
/// Processes batches of inputs through a layer of weights
// Functionally replaces matrix product -- no transform or reformatting needed
fn batch_product(input_batch: &Vec<Vec<f32>>, weight_set: &Vec<Vec<f32>>, biases: &Vec<f32>) -> Vec<Vec<f32>> {
    assert_eq![input_batch[0].len(), weight_set[0].len()];
    let mut product: Vec<Vec<f32>> = Vec::new();
    for inputs in input_batch.iter() {
        let mut sample: Vec<f32> = Vec::new();
        for (w, weights) in weight_set.iter().enumerate() {
            sample.push(inputs.dot_product(&weights) + biases[w]);
        }
        product.push(sample);
    }
    return product
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

fn two_layer_batches() {
    let input_batch: Vec<Vec<f32>> = vec![
         vec![1., 2., 3., 4.],
         vec![1., 2., 3., 4.],
         vec![1., 2., 3., 4.],
    ]; 
    let weight_set0: Vec<Vec<f32>> = vec![
        vec![0.1, 0.2, 0.3, 0.4],
        vec![0.1, 0.2, 0.3, 0.4],
        vec![0.1, 0.2, 0.3, 0.4],
    ];
    // Must match the output dimensions from layer0
    let weight_set1: Vec<Vec<f32>> = vec![
        vec![0.1, 0.2, 0.3],
        vec![0.1, 0.2, 0.3],
        vec![0.1, 0.2, 0.3],
    ];
    let biases0: Vec<f32> = vec![2., 2., 2.];
    let biases1: Vec<f32> = vec![2., 2., 2.];
    
    let layer0_outputs: Vec<Vec<f32>> = batch_product(&input_batch, &weight_set0, &biases0);
    let layer1_outputs: Vec<Vec<f32>> = batch_product(&layer0_outputs, &weight_set1, &biases1);

    for outputs in layer1_outputs {
        for output in outputs { print!("{output} ") };
        print!("\n");
    }
}

fn one_layer_batches() {
    let input_batch: Vec<Vec<f32>> = vec![
         vec![1., 2., 3., 4.],
         vec![1., 2., 3., 4.],
         vec![1., 2., 3., 4.],
    ]; 
    let weight_set: Vec<Vec<f32>> = vec![
        vec![0.1, 0.2, 0.3, 0.4],
        vec![0.1, 0.2, 0.3, 0.4],
        vec![0.1, 0.2, 0.3, 0.4],
    ];
    let biases: Vec<f32> = vec![2., 2., 2.];
    let output_set: Vec<Vec<f32>> = batch_product(&input_batch, &weight_set, &biases);
    for outputs in output_set {
        for output in outputs { print!("{output} ") };
        print!("\n");
    }
}

fn one_layer_clean() {
    let inputs: Vec<f32> = vec![1., 2., 3.];
    let weight_set: Vec<Vec<f32>> = vec![
        vec![0.1, 0.2, 0.3],
        vec![0.1, 0.2, 0.3],
        vec![0.1, 0.2, 0.3],
    ];
    let biases: Vec<f32> = vec![2., 2., 2.];
    let outputs: Vec<f32> = weight_set.iter()
                                      .enumerate()
                                      .map(|(w, weights)| { inputs.dot_product(&weights) + biases[w] })
                                      .collect::<Vec<f32>>();
    
    for output in outputs { print!("{output} ") };
    print!("\n");
}

fn one_neuron_clean() {
    let inputs: Vec<f32> = vec![1., 2., 3.];
    let weights: Vec<f32> = vec![0.1, 0.2, 0.3];
    let bias: f32 = 2.;
    let output: f32 = inputs.dot_product(&weights) + bias;
    println!("{output}");
}

fn one_layer_fire() {
    
    let inputs: Vec<f32> = vec![1., 2., 3.];

    let weights0: Vec<f32> = vec![0.1, 0.2, 0.3];
    let weights1: Vec<f32> = vec![0.1, 0.2, 0.3];
    let weights2: Vec<f32> = vec![0.1, 0.2, 0.3];
    let weight_set: Vec<Vec<f32>> = vec![weights0, weights1, weights2];
    
    let bias0: f32 = 2.;
    let bias1: f32 = 2.;
    let bias2: f32 = 2.;
    let biases: Vec<f32> = vec![bias0, bias1, bias2];

    let outputs: Vec<f32> = weight_set.iter()
                                      .enumerate()
                                      .map(|(w, weights)| {
                                           inputs.iter()
                                               .enumerate()
                                               .map(|(i, input)| input * weights[i])
                                               .collect::<Vec<f32>>().into_iter()
                                               .sum::<f32>() + biases[w]
                                      }).collect::<Vec<f32>>();

    for output in outputs { print!("{output} ") };
    print!("\n");
}

fn one_neuron_fire() {
    let inputs: Vec<f32> = vec![1., 2., 3.];
    let weights: Vec<f32> = vec![0.1, 0.2, 0.3];
    let bias: f32 = 2.;

    let output: f32 = inputs.iter()
                            .enumerate()
                            .map(|(i, input)| input*weights[i])
                            .collect::<Vec<f32>>().into_iter()
                            .sum::<f32>() + bias;

    println!("{output}");
}
