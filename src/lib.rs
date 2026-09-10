//! ArcFace w600k_r50 inference. Aligned inputs are 112×112 uint8 BGR;
//! embeddings are 512 unnormalized float32 values, as in InsightFace.
pub mod alignment;
mod cnn;
mod engine;
mod model;
mod onnx;
mod plan;
use anyhow::{Result, ensure};
use std::path::Path;
pub const SIZE: usize = 112;
pub const EMBEDDING: usize = 512;
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub device: i32,
    pub max_batch: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            device: 0,
            max_batch: 16,
        }
    }
}
/// A resident model and its exclusive execution stream.
pub struct ArcFace {
    cnn: cnn::Cnn,
}
impl ArcFace {
    /// Validate and pack the model, compile kernels, and allocate resident storage.
    pub fn load(path: impl AsRef<Path>, options: Options) -> Result<Self> {
        ensure!(
            (1..=64).contains(&options.max_batch),
            "max_batch must be 1..=64"
        );
        Ok(Self {
            cnn: cnn::Cnn::new(
                model::load(path.as_ref())?,
                options.device,
                options.max_batch,
            )?,
        })
    }
    /// Encode a contiguous batch of aligned 112×112 BGR crops, chunking as needed.
    pub fn embeddings(&mut self, crops: &[u8]) -> Result<Vec<[f32; EMBEDDING]>> {
        ensure!(
            crops.len().is_multiple_of(SIZE * SIZE * 3),
            "expected complete 112×112 BGR crops"
        );
        let mut out = vec![[0.; EMBEDDING]; crops.len() / (SIZE * SIZE * 3)];
        for (input, out) in crops
            .chunks(self.cnn.max_batch * SIZE * SIZE * 3)
            .zip(out.chunks_mut(self.cnn.max_batch))
        {
            self.cnn.run(input)?;
            self.cnn
                .engine
                .read(self.cnn.outputs[0], bytemuck::cast_slice_mut(out))?;
        }
        Ok(out)
    }
    /// Compare resident graph replay and direct dispatch, excluding crop transfers.
    pub fn benchmark(&mut self, crops: &[u8], samples: usize) -> Result<ForwardTimings> {
        ensure!(
            !crops.is_empty() && crops.len() <= self.cnn.max_batch * SIZE * SIZE * 3,
            "benchmark requires one nonempty batch"
        );
        self.embeddings(crops)?;
        self.cnn
            .engine
            .benchmark(crops.len() / (SIZE * SIZE * 3), samples)
    }
    /// Align and embed faces from a packed BGR image and five landmarks per face.
    pub fn embed(
        &mut self,
        image: &[u8],
        width: usize,
        height: usize,
        landmarks: &[[[f32; 2]; 5]],
    ) -> Result<Vec<[f32; EMBEDDING]>> {
        let mut crops = Vec::new();
        for points in landmarks {
            crops.extend(alignment::crop(image, width, height, points)?);
        }
        self.embeddings(&crops)
    }
}
/// Cosine similarity; rejects non-finite or zero-length embeddings.
pub fn similarity(a: &[f32; EMBEDDING], b: &[f32; EMBEDDING]) -> Result<f64> {
    ensure!(
        a.iter().chain(b).all(|v| v.is_finite()),
        "non-finite embedding"
    );
    let dot = a
        .iter()
        .zip(b)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum::<f64>();
    let norm = |a: &[f32]| a.iter().map(|a| (*a as f64).powi(2)).sum::<f64>();
    let d = (norm(a) * norm(b)).sqrt();
    ensure!(d > 0., "zero embedding");
    Ok(dot / d)
}

#[cfg(test)]
mod tests;

pub use engine::{Distribution, ForwardTimings};
