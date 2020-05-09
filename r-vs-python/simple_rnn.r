require(dplyr)
require(keras)
require(progress)
require(readtext)
require(tokenizers)
require(yaml)
rm(list=ls())

# --- script settings start ---
document <- '../data/marktwain.txt'
yml <- './simple_rnn.yml'
save_path <- '../data/simple_rnn.r.model'
# --- script settings end ---

FLAGS <- yaml.load_file(yml)

tokens <- readtext(document) %>% tokenize_words(simplify = T)
unique_tokens <- tokens %>% unique()
rm(document, yml)

samples <- array(dim = c(length(tokens) - FLAGS$sample_length + 1, FLAGS$sample_length))
for(i in 1:(dim(samples)[1])) {
  samples[i, ] <- tokens[i:(i + FLAGS$sample_length - 1)]
}
rm(tokens, i)

# build a simple model
# words -> one-hot -> rnn -> dense -> output
model <-
  keras_model_sequential() %>%
  layer_simple_rnn(
    units = FLAGS$units,
    input_shape = c(FLAGS$sample_length - 1, length(unique_tokens))) %>%
  layer_dense(length(unique_tokens)) %>%
  layer_activation("softmax")
compile(
  model,
  loss = "categorical_crossentropy", 
  optimizer = optimizer_nadam())

make_batch <- function(samples, batch_start, batch_size) {
  sz = dim(samples)
  batch <- batch_start:min(sz[1], batch_start + batch_size - 1)
  batch_x <- samples[batch,-sz[2]]
  batch_y <- samples[batch, sz[2]]
  list(x = batch_x, y = batch_y)
}
one_hot_batch <- function(batch, unique_tokens) {
  one_hot_x <- array(0, dim = c(dim(batch$x), length(unique_tokens)))
  one_hot_y <- array(0, dim = c(length(batch$y), length(unique_tokens)))
  sz = dim(batch$x)
  for(i in 1:sz[1]) {
    for(j in 1:sz[2]) {
      one_hot_x[i,j,] <- as.integer(unique_tokens == batch$x[i,j])
    }
    one_hot_y[i,] <- as.integer(unique_tokens == batch$y[i])
  }
  list(x = one_hot_x, y = one_hot_y)
}

# train the model ---
t1 <- Sys.time()
sz = dim(samples)
for(i in 1:FLAGS$epochs) {
  batch_i <- 1
  pb <-
    progress_bar$new(
      format = 'batch :current/:total [:bar] eta: :eta',
      total = ceiling(sz[1]/FLAGS$batch_size))
  pb$tick(0)
  while(batch_i <= sz[1]) {
    batch <- make_batch(samples, batch_i, FLAGS$batch_size)
    one_hot <- one_hot_batch(batch, unique_tokens)
    loss = train_on_batch(model, one_hot$x, one_hot$y)
    rm(batch, one_hot)
    batch_i <- batch_i + FLAGS$batch_size
    pb$tick()
  }
  print(sprintf('Epoch: %i, Loss: %f', i, loss))
  rm(batch_i, pb, loss)
}
t2 <- Sys.time()
print(t2 - t1)

rm(model, save_path, make_batch, one_hot_batch, t1, t2, sz, i)
