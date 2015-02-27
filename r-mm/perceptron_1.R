# Copyright (c) 2015 Alexey Kodanev <akodanev@gmail.com>
# All Rights Reserved.
#
# Single-layer artificial neural network with learning
# ====================================================

source("activation_fn.R");
source("plot_data.R");

source("ex_data_img16x16.R");

single_net_calc = function(b, w, x, n_cnt, k)
{
  s = rep(0, n_cnt);

  for (i in seq(1, n_cnt)) {
    s[i] = b[i] + w[2,i, ] %*% x;
  }

  #y = act_threshold_fn(s);
  y = act_sigmoid_fn(s, k);

  return (y);
}

single_net_learn = function(b, w, y, d, x, n_cnt)
{
  # Learning...
  
  # learning rate 0.1..0.9
  lr = 1;
  
  # delta rule, h is the weighted sum of the neuron's inputs
  # h = rep(0, n_cnt);
  
  n_err = lr * (d - y);
  
  for (i in seq(1, n_cnt)) {
    w[2,i,] = w[2,i,] + n_err[i] * x;
  }

  return (w);
}

single_net = function()
{
  ret = ex_data_for_learn();
  ex_x = ret[[1]]; ex_d = ret[[2]];
  rm(ret);

  # inputs
  i_cnt = length(ex_x[1,]);
  # neurons in the layer
  n_cnt = length(ex_d[1,]);

  fn_k = 1;

  b = seq(0, n_cnt);
  # desired output, for learning
  d = rep(0, n_cnt);

  # weights to remember
  w_mem_step = 3;

  # initialize weights
  w = array(0, dim = c(w_mem_step, n_cnt, i_cnt));

  # run auto training
  i = 0;
  while (i < 20) {

    id = (i %% n_cnt) + 1;
    x = ex_x[id,];
    d = ex_d[id,];

    y = single_net_calc(b, w, x, n_cnt, fn_k);
    w = single_net_learn(b, w, y, d, x, n_cnt);

    i = i + 1;
  }

  plot_weights(w[2,,]);

  # test run
  for (i in seq(1, 5)) {

    cat("\nTest #", i, ": ");

    x = ex_data_for_test(i);

    y = single_net_calc(b, w, x, n_cnt, fn_k);

    ex_analyze_results(y);
  }

}
