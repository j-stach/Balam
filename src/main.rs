// When optimizing performance, implement ndarray's dot or add rayon par_iter to the dot trait below
// And change matrix to use arrays instead of vectors

use rand::Rng;

fn main() {

    two_dense_layer();

}

fn two_dense_layer() {

    let sample_batch: Vec<Vec<f32>> = vec![
         vec![1., 2., 3., 4.],
         vec![1., 2., 3., 4.],
         vec![1., 2., 3., 4.],
    ]; 
    
    let layer0 = DenseLayer::new(4, 3);
    println!("Hidden Layer:");
    for node in layer0.0.iter() {
        println!("bias: {}", node.bias);
        print!("weights: ");
        for weight in &node.weights {
            print!("{weight} ");
        }
        print!("\n");
    }
    let layer1 = DenseLayer::new(3, 3);
    println!("Output Layer:");
    for node in layer1.0.iter() {
        println!("bias: {}", node.bias);
        print!("weights: ");
        for weight in &node.weights {
            print!("{weight} ");
        }
        print!("\n");
    }

    let layer0_outputs: Vec<Vec<f32>> = layer0.forward(&sample_batch);
    let layer1_outputs: Vec<Vec<f32>> = layer1.predict(&layer0_outputs);

    println!("Outputs:");
    for outputs in &layer1_outputs { 
        for output in outputs { print!("{output} ") };
        print!("\n");
    };

    let class = layer1_outputs.classify();

    println!("Class:");
    for result in &class { println!("{result} ") };
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
    fn argmax(&self) -> usize;
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
    /// Find the index of the greatest f32 in the vector
    // If you're looking for a max when your vec is empty, your program deserves to crash.
    fn argmax(&self) -> usize {
        let argmax = self.iter()
                         .enumerate()
                         .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
                         .map(|(i, _)| i).unwrap();
        return argmax
    }
}

trait Output {
    fn classify(&self) -> Vec<usize>;
    //fn cce_loss() -> 
}
impl Output for Vec<Vec<f32>> {
    /// Determines a result from a probability distribution
    fn classify(&self) -> Vec<usize> {
        let classification = self.iter().map(|s| s.argmax()).collect::<Vec<usize>>();
        return classification
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
    /// Output pass that returns a probability distribution 
    fn predict(&self, input_batch: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
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







/// Activation patterns
// split these off to an "activation function" crate
trait Activation {
    fn relu(&self) -> Vec<Vec<f32>>;
    fn softmax(&self) -> Vec<Vec<f32>>;
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
    // Exploding value risk still needs to be handled? (Only if input values are large?)
    fn softmax(&self) -> Vec<Vec<f32>> {
        let e: f32 = 2.718282;
        // Sum each sample after applying exponents
        let output_sum: Vec<f32> = self.iter()
                                  .map(|os| { 
                                      os.iter()
                                        .map(|o| {
                                            e ** o
                                        }).collect::<Vec<f32>>().sum()
                                  }).collect::<Vec<f32>>();
        // Run it again to generate probability distribution
        let outputs: Vec<Vec<f32>> = self.iter()
                                         .enumerate()
                                         .map(|(i, os)| { 
                                             os.iter()
                                               .map(|o| {
                                                   (e ** o) / output_sum[i]
                                               }).collect::<Vec<f32>>()
                                         }).collect::<Vec<Vec<f32>>>();
        return outputs
    }
}







