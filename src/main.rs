// When optimizing performance, implement ndarray's dot or add rayon par_iter to the dot trait below
// And change matrix to use arrays instead of vectors

use rand::Rng;

fn main() {

    one_dense_layer();

}

fn one_dense_layer() {

    let layer = DenseLayer::new(4, 3);
    for node in layer.0.iter() {
        println!("bias: {}", node.bias);
        print!("weights: ");
        for weight in &node.weights {
            print!("{weight} ");
        }
        print!("\n");
    }

    let input_batch: Vec<Vec<f32>> = vec![
         vec![1., 2., 3., 4.],
         vec![1., 2., 3., 4.],
         vec![1., 2., 3., 4.],
    ]; 
    
    let layer0_outputs: Vec<Vec<f32>> = layer.forward(&input_batch);
    for outputs in layer0_outputs {
        for output in outputs { print!("{output} ") };
        print!("\n");
    }

}

/// Trims away negative values for ReLU activation
fn take_pos(val: f32) -> f32 {
    let mut val = val;
    if val < 0. { val = 0.; }
    return val
}

/// Math functions for f32 vecs
// include take_pos when moving to new mod
trait MathUtils {
    fn dot_product(&self, rhs: &Vec<f32>) -> f32;
    fn sum(&self) -> f32;
}
impl MathUtils for Vec<f32> {
    /// Dot product of two vectors of f32
    fn dot_product(&self, rhs: &Vec<f32>) -> f32 {
        assert_eq![self.len(), rhs.len()];
        let mut product: f32 = 0.;
        for i in 0..self.len() {
            product += self[i] * rhs[i];
        }
        return product
    }
    /// Add together all elements of a vector of f32
    fn sum(&self) -> f32 {
        let mut sum: f32 = 0.;
        for f in self { sum += f };
        return sum
    }
}

/// "Neuron" node
struct Node {
    weights: Vec<f32>,
    bias: f32,
}
/// Basic layer of neural nodes
// Eventually, make this an enum?
struct DenseLayer (Vec<Node>);
impl DenseLayer {
    /// Create a new layer of fully connected nodes
    fn new(input_count: usize, node_count: usize) -> Self {
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
    fn forward(&self, input_batch: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
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
}

/// Activation patterns
// split these off to an "activation function" crate
trait Activation {
    fn relu(&self) -> Vec<Vec<f32>>;
    fn softmax(&self) -> Vec<Vec<f32>>;
    //fn linear(&self) -> Vec<Vec<f32>> {}
}
impl Activation for Vec<Vec<f32>> {
    /// Rectified linear activation pattern, for hidden layers
    fn relu(&self) -> Vec<Vec<f32>> {
        self.iter()
            .map(|os| { 
                os.iter()
                  .map(|o| {
                        take_pos(*o)
                  }).collect::<Vec<f32>>()
            }).collect::<Vec<Vec<f32>>>()
    }
    /// Softmax exponential activation function, for the output layer in classification models
    fn softmax(&self) -> Vec<Vec<f32>> {
        let e: f32 = 2.718282;
        let output_sum: f32 = self.iter()
                                  .map(|os| { 
                                      os.iter()
                                        .map(|o| {
                                            e ** o
                                        }).collect::<Vec<f32>>().sum()
                                  }).collect::<Vec<f32>>().sum();

        let outputs: Vec<Vec<f32>> = self.iter()
                                         .map(|os| { 
                                             os.iter()
                                               .map(|o| {
                                                   (e ** o) / output_sum
                                               }).collect::<Vec<f32>>()
                                         }).collect::<Vec<Vec<f32>>>();
        return outputs

    }
}

