use ndarray::{Array1, Array2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
#[derive(Clone, Debug)]
pub struct NeuronStats {
    pub mean_activation: f32,
    pub peak_activation: f32,
    pub active_ratio: f32,
    pub last_update: usize,
}

#[pymethods]
impl NeuronStats {
    #[new]
    fn new() -> Self {
        Self {
            mean_activation: 0.0,
            peak_activation: 0.0,
            active_ratio: 1.0,
            last_update: 0,
        }
    }
}

#[pyclass]
pub struct DynamicOptimizer {
    activation_threshold: f32,
    min_active_ratio: f32,
    update_frequency: usize,
    ema_alpha: f32,
    step_counter: usize,
    neuron_stats: HashMap<String, Vec<NeuronStats>>,
}

#[pymethods]
impl DynamicOptimizer {
    #[new]
    #[pyo3(signature = (
        activation_threshold = 0.1,
        min_active_ratio = 0.3,
        update_frequency = 10,
        ema_alpha = 0.1
    ))]
    pub fn new(
        activation_threshold: f32,
        min_active_ratio: f32,
        update_frequency: usize,
        ema_alpha: f32,
    ) -> Self {
        Self {
            activation_threshold,
            min_active_ratio,
            update_frequency,
            ema_alpha,
            step_counter: 0,
            neuron_stats: HashMap::new(),
        }
    }

    pub fn update_activations(&mut self, layer_name: &str, activations: &PyArray2<f32>) -> PyResult<()> {
        if self.step_counter % self.update_frequency != 0 {
            self.step_counter += 1;
            return Ok(());
        }

        let arr = unsafe { activations.as_array() };
        let neuron_means = arr.mean_axis(Axis(0)).unwrap();
        let neuron_peaks = arr.fold_axis(Axis(0), 0.0f32, |&acc, &x| acc.max(x.abs()));

        if !self.neuron_stats.contains_key(layer_name) {
            let num_neurons = neuron_means.len();
            self.neuron_stats.insert(
                layer_name.to_string(),
                vec![NeuronStats::new(); num_neurons],
            );
        }

        let stats = self.neuron_stats.get_mut(layer_name).unwrap();
        for (i, (mean, &peak)) in neuron_means.iter().zip(neuron_peaks.iter()).enumerate() {
            stats[i].mean_activation = stats[i].mean_activation * (1.0 - self.ema_alpha)
                + mean.abs() * self.ema_alpha;
            stats[i].peak_activation = stats[i].peak_activation.max(peak);
            
            let is_active = stats[i].mean_activation >= self.activation_threshold
                || stats[i].peak_activation >= self.activation_threshold * 2.0;
            stats[i].active_ratio = stats[i].active_ratio * (1.0 - self.ema_alpha)
                + if is_active { 1.0 } else { 0.0 } * self.ema_alpha;
            stats[i].last_update = self.step_counter;
        }

        self.step_counter += 1;
        Ok(())
    }

    pub fn get_active_neurons(&self, layer_name: &str) -> PyResult<Vec<usize>> {
        let stats = self.neuron_stats.get(layer_name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Layer not found")
        })?;

        let mut active_neurons = Vec::new();
        let total_neurons = stats.len();
        let min_neurons = (total_neurons as f32 * self.min_active_ratio).ceil() as usize;

        for (i, stat) in stats.iter().enumerate() {
            if stat.active_ratio >= self.min_active_ratio {
                active_neurons.push(i);
            }
        }

        if active_neurons.len() < min_neurons {
            let mut neuron_scores: Vec<(usize, f32)> = stats
                .iter()
                .enumerate()
                .map(|(i, s)| (i, s.mean_activation))
                .collect();
            neuron_scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            for (i, _) in neuron_scores.iter().take(min_neurons) {
                if !active_neurons.contains(i) {
                    active_neurons.push(*i);
                }
            }
        }

        Ok(active_neurons)
    }

    #[pyo3(name = "optimize_layer")]
    pub fn optimize_layer_py(
        &self,
        layer_name: &str,
        weights: &PyArray2<f32>,
        bias: Option<&PyArray1<f32>>,
    ) -> PyResult<(Py<PyArray2<f32>>, Option<Py<PyArray1<f32>>>)> {
        let weights_arr = unsafe { weights.as_array() };
        let bias_arr = bias.map(|b| unsafe { b.as_array() });

        let active_neurons = self.get_active_neurons(layer_name)?;
        let in_features = weights_arr.shape()[1];
        
        let mut optimized_weights = Array2::zeros((active_neurons.len(), in_features));
        for (new_idx, &old_idx) in active_neurons.iter().enumerate() {
            optimized_weights.row_mut(new_idx).assign(&weights_arr.row(old_idx));
        }

        let optimized_bias = bias_arr.map(|b| {
            let mut new_bias = Array1::zeros(active_neurons.len());
            for (new_idx, &old_idx) in active_neurons.iter().enumerate() {
                new_bias[new_idx] = b[old_idx];
            }
            new_bias
        });

        Python::with_gil(|py| {
            Ok((
                optimized_weights.into_pyarray(py).to_owned(),
                optimized_bias.map(|b| b.into_pyarray(py).to_owned()),
            ))
        })
    }

    pub fn get_stats(&self, layer_name: &str) -> PyResult<(usize, usize, f32)> {
        let stats = self.neuron_stats.get(layer_name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Layer not found")
        })?;

        let total_neurons = stats.len();
        let active_neurons = self.get_active_neurons(layer_name)?.len();
        let compression_ratio = 1.0 - (active_neurons as f32 / total_neurons as f32);

        Ok((total_neurons, active_neurons, compression_ratio))
    }
} 