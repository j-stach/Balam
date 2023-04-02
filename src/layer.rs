
use rand::Rng;

use crate::activation::*; 
use crate::utils::*;

/// "Neuron" node
pub struct Node {
    pub weights: Vec<f32>,
    pub bias: f32,
}

/// Basic layer of neural nodes
// Eventually, make this an enum?
pub struct DenseLayer (pub Vec<Node>);
impl DenseLayer {
    /// Create a new layer of fully connected nodes
    pub fn new(input_count: usize, node_count: usize) -> Self {
        let mut layer: Vec<Node> = Vec::new();
        for _ in 0..node_count {
            let mut node = Node {
                weights: Vec::with_capacity(input_count), 
                bias: 0.,
            };
            for _ in 0..input_count {
                node.weights.push(rand::thread_rng().gen::<f32>());
            }
            layer.push(node);
        }
        return DenseLayer(layer)
    }

    /// Forward pass of batches of inputs through a layer of weights
    pub fn forward(&self, input_batch: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let weight_set: Vec<Vec<f32>> = self.0.iter().map(|n| n.weights.clone()).collect::<Vec<Vec<f32>>>();
        let biases: Vec<f32> = self.0.iter().map(|n| n.bias.clone()).collect::<Vec<f32>>();
        assert_eq![input_batch[0].len(), weight_set[0].len()];
        let mut product: Vec<Vec<f32>> = Vec::new();
        for inputs in input_batch.iter() {
            let mut sample: Vec<f32> = Vec::new();
            for (w, weights) in weight_set.iter().enumerate() {
                sample.push(inputs.dot_product(&weights) + biases[w]);
            }
            product.push(sample);
        }
        let product = product.relu();
        return product
    }
    /// Output pass that returns a probability distribution 
    pub fn predict(&self, input_batch: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let weight_set: Vec<Vec<f32>> = self.0.iter().map(|n| n.weights.clone()).collect::<Vec<Vec<f32>>>();
        let biases: Vec<f32> = self.0.iter().map(|n| n.bias.clone()).collect::<Vec<f32>>();
        assert_eq![input_batch[0].len(), weight_set[0].len()];
        let mut product: Vec<Vec<f32>> = Vec::new();
        for inputs in input_batch.iter() {
            let mut sample: Vec<f32> = Vec::new();
            for (w, weights) in weight_set.iter().enumerate() {
                sample.push(inputs.dot_product(&weights) + biases[w]);
            }
            product.push(sample);
        }
        let product = product.softmax();
        return product
    }
}    