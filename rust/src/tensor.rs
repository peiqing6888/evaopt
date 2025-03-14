use ndarray::{ArrayD, Array1};
use num_traits::Float;

/// Tensor data types supported by the optimizer
#[derive(Debug, Clone, Copy)]
pub enum TensorType {
    F32,
    I8,
    I4,
}

impl TensorType {
    /// Convert tensor to F32
    pub fn to_f32(&self) -> ArrayD<f32> {
        match self {
            TensorType::F32 => ArrayD::zeros(vec![0]),
            TensorType::I8 => ArrayD::zeros(vec![0]),
            TensorType::I4 => ArrayD::zeros(vec![0]),
        }
    }
    
    /// Convert tensor to specified type
    pub fn convert_to(&self, target_type: &str) -> TensorType {
        match target_type {
            "f32" => TensorType::F32,
            "i8" => TensorType::I8,
            "i4" => TensorType::I4,
            _ => *self,
        }
    }
    
    /// Get memory usage in bytes
    pub fn memory_size(&self) -> usize {
        match self {
            TensorType::F32 => 4,
            TensorType::I8 => 1,
            TensorType::I4 => 1,
        }
    }
    
    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        &[]
    }
}

/// Tensor operations trait
pub trait TensorOps {
    fn sparsify(&mut self, threshold: f32);
    fn quantize(&mut self, bits: u8) -> Vec<u8>;
    fn compress(&mut self) -> Vec<u8>;
}

impl TensorOps for ArrayD<f32> {
    fn sparsify(&mut self, threshold: f32) {
        self.mapv_inplace(|x| if x.abs() < threshold { 0.0 } else { x });
    }
    
    fn quantize(&mut self, bits: u8) -> Vec<u8> {
        let scale = (2.0f32.powi(bits as i32) - 1.0) / self.fold(0.0f32, |m, &x| m.max(x.abs()));
        let quantized: Vec<u8> = self.mapv(|x| (x * scale) as u8).into_raw_vec();
        quantized
    }
    
    fn compress(&mut self) -> Vec<u8> {
        let mut compressed = Vec::new();
        let mut count = 0u8;
        let mut last_val = 0.0f32;
        
        for &val in self.iter() {
            if val == last_val && count < 255 {
                count += 1;
            } else {
                compressed.extend_from_slice(&last_val.to_le_bytes());
                compressed.push(count);
                last_val = val;
                count = 1;
            }
        }
        
        if count > 0 {
            compressed.extend_from_slice(&last_val.to_le_bytes());
            compressed.push(count);
        }
        
        compressed
    }
} 