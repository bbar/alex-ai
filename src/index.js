const tf = require('@tensorflow/tfjs-node');
const math = require('mathjs');

function standardize(featureSet) {
  const stds = math.std(featureSet, 0);
  const means = math.mean(featureSet, 0);

  return featureSet.map((features) => {
    return features.map((feature, j) => {
      const std = stds[j];
      const mean = means[j];
      return (feature - mean) / std;
    });
  });
}

(async () => {
  const features = [
    [0, 0],
    [0.5, 0.5],
    [1, 1],
  ];

  const labels = [
    [3],
    [6],
    [9],
  ];

  // Standardize features
  const _features = standardize(features);

  const model = tf.sequential({
    layers: [
      tf.layers.dense({
        inputShape: [2],
        units: 4,
        activation: 'sigmoid',
      }),
      tf.layers.dense({
        units: 500,
        activation: 'relu',
      }),
      tf.layers.dense({
        units: 1,
        activation: 'relu',
      }),
    ],
  });

  model.compile({
    optimizer: tf.train.sgd(0.1),
    loss: tf.losses.meanSquaredError,
  });

  const xs = tf.tensor(_features);
  const ys = tf.tensor(labels);

  const history = await model.fit(xs, ys, {
    epochs: 2000,
    verbose: 1,
    shuffle: true,
  });

  const prediction = model.predict(xs);
  prediction.print();
})();


