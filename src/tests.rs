use super::*;
#[test]
fn input_and_alignment_validation() {
    assert!(alignment::transform(&[[0.; 2]; 5]).is_err());
    assert!(alignment::transform(&[[f32::NAN; 2]; 5]).is_err());
    assert!(alignment::crop(&[], 112, 112, &alignment::TEMPLATE).is_err());
    assert!(similarity(&[0.; 512], &[1.; 512]).is_err());
    assert!((similarity(&[1.; 512], &[1.; 512]).unwrap() - 1.).abs() < 1e-12);
    let t = alignment::transform(&alignment::TEMPLATE).unwrap();
    for (row, want) in t.iter().zip([[1., 0., 0.], [0., 1., 0.]]) {
        for (v, w) in row.iter().zip(want) {
            assert!((v - w).abs() < 1e-4);
        }
    }
}
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot = a
        .iter()
        .zip(b)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum::<f64>();
    let norm = |a: &[f32]| a.iter().map(|a| (*a as f64).powi(2)).sum::<f64>();
    dot / (norm(a) * norm(b)).sqrt()
}
#[test]
#[ignore = "requires ARCFACE_MODEL and gfx1151"]
fn native_reference_and_replay() -> Result<()> {
    use tract_onnx::prelude::*;
    let path = std::env::var("ARCFACE_MODEL")?;
    let mut model = ArcFace::load(
        &path,
        Options {
            device: 0,
            max_batch: 4,
        },
    )?;
    let reference = tract_onnx::onnx()
        .model_for_path(&path)?
        .with_input_fact(0, f32::fact([1, 3, 112, 112]).into())?
        .into_optimized()?
        .into_runnable()?;
    let crops: Vec<u8> = (0..4 * 112 * 112 * 3)
        .map(|i| ((i * 13 + i / 379) % 256) as u8)
        .collect();
    let batch = model.embeddings(&crops)?;
    for (i, crop) in crops.chunks(112 * 112 * 3).enumerate() {
        let blob = tract_ndarray::Array4::from_shape_fn((1, 3, 112, 112), |(_, c, y, x)| {
            (crop[(y * 112 + x) * 3 + 2 - c] as f32 - 127.5) / 127.5
        });
        let expected = reference.run(tvec![blob.into_tensor().into()])?;
        let data = expected[0].to_array_view::<f32>()?;
        let want = data.as_slice().unwrap();
        assert!(
            cosine(&batch[i], want) > 0.99995,
            "image {i} cosine {}",
            cosine(&batch[i], want)
        );
        assert_eq!(model.embeddings(crop)?[0], batch[i]);
    }
    assert_eq!(model.embeddings(&crops)?, batch);
    assert!(model.embeddings(&[0; 5]).is_err());
    Ok(())
}
#[test]
#[ignore = "requires ARCFACE_MODEL; CPU model import"]
fn importer_liveness() -> Result<()> {
    let p = model::load(std::path::Path::new(&std::env::var("ARCFACE_MODEL")?))?;
    assert_eq!(p.ops.len(), 56);
    assert_eq!(p.buffers, [200704, 1605632, 1605632, 401408]);
    for l in p.ops {
        assert_ne!(l.src_buf, l.dst_buf);
        assert_ne!(l.extra_buf, l.dst_buf);
    }
    Ok(())
}
#[test]
fn malformed_onnx_returns_errors() {
    use onnx_protobuf::{GraphProto, Message, ModelProto, NodeProto, ValueInfoProto};
    assert!(onnx::Network::from_bytes(&[], 112).is_err());
    assert!(onnx::Network::from_bytes(&[0x3a, 0xff], 112).is_err());
    let node = NodeProto {
        op_type: "Conv".into(),
        input: vec!["x".into()],
        output: vec!["y".into()],
        ..Default::default()
    };
    let graph = GraphProto {
        node: vec![node],
        input: vec![ValueInfoProto {
            name: "x".into(),
            ..Default::default()
        }],
        ..Default::default()
    };
    let model = ModelProto {
        graph: Some(graph).into(),
        ..Default::default()
    };
    assert!(onnx::Network::from_bytes(&model.write_to_bytes().unwrap(), 112).is_err());
}
#[test]
fn landmark_fixture() -> Result<()> {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../tests/fixtures/t1_arcface.json"))?;
    for face in fixture["faces"].as_array().unwrap() {
        let points: [[f32; 2]; 5] = serde_json::from_value(face["kps"].clone())?;
        let want: [[f64; 3]; 2] = serde_json::from_value(face["M"].clone())?;
        let got = alignment::transform(&points)?;
        for (a, b) in got.iter().flatten().zip(want.iter().flatten()) {
            assert!((a - b).abs() < 2e-4, "alignment delta {}", (a - b).abs());
        }
    }
    Ok(())
}
#[test]
#[ignore = "requires ARCFACE_MODEL, bundled InsightFace fixture and gfx1151"]
fn insightface_fixture() -> Result<()> {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../tests/fixtures/t1_arcface.json"))?;
    let image = image::load_from_memory(include_bytes!("../tests/fixtures/t1.png"))?.to_rgb8();
    let (w, h) = image.dimensions();
    let mut bgr = image.into_raw();
    for p in bgr.chunks_exact_mut(3) {
        p.swap(0, 2);
    }
    let mut model = ArcFace::load(std::env::var("ARCFACE_MODEL")?, Options::default())?;
    let mut expected = vec![];
    let mut landmarks = vec![];
    for face in fixture["faces"].as_array().unwrap() {
        landmarks.push(serde_json::from_value::<[[f32; 2]; 5]>(
            face["kps"].clone(),
        )?);
        expected.push(serde_json::from_value::<Vec<f32>>(
            face["embedding"].clone(),
        )?);
    }
    let got = model.embed(&bgr, w as usize, h as usize, &landmarks)?;
    for (a, b) in got.iter().zip(&expected) {
        assert!(cosine(a, b) > 0.99995, "cosine {}", cosine(a, b));
    }
    for i in 0..got.len() {
        for j in 0..got.len() {
            assert!((cosine(&got[i], &got[j]) - cosine(&expected[i], &expected[j])).abs() < 0.001);
        }
    }
    Ok(())
}
