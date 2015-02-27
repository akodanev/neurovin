# Copyright (c) 2015 Alexey Kodanev <akodanev@gmail.com>
# All Rights Reserved.
#
# Multi-layer artificial neural network with learning (+1 hidden layer)
# ===================================================================

source("activation_fn.R");
source("plot_data.R");

source("ex_data_img16x16.R");

hy = 0;
hs = 0;
s = 0;
w = 0;
wh = 0;
b = 0;
bh = 0;
x = 0; hy = 0; y = 0;

multi_net_calc = function(h_cnt, n_cnt, k)
{
  hs = rep(0, h_cnt);
  for (i in seq(1, h_cnt)) {
    hs[i] = bh[i] + wh[2,i, ] %*% x;
  }

  hy = act_sigmoid_fn(hs, k);

  s = rep(0, n_cnt);
  for (i in seq(1, n_cnt)) {
    s[i] = b[i] + w[2,i, ] %*% hy;
  }

  y = act_sigmoid_fn(s, k);
}

multi_net_learn = function(d, h_cnt, n_cnt)
{
  # Learning...
  # learning rate 0.1..0.9
  lr = 1;

  do = lr * act_sigmoid_deriv_fn(s, k) * (d - y);

  dh = act_sigmoid_deriv_fn(hs, k) * (do %*% w[2,i,]);

  for (i in seq(1, n_cnt)) {
    w[2,i,] = w[2,i,] + do[i] * y;
  }

  for (i in seq(1, h_cnt)) {
    wh[2,i,] = wh[2,i,] + dh[i] * x;
  }
}

multi_net = function()
{
  ret = ex_data_for_learn();
  ex_x = ret[[1]]; ex_d = ret[[2]];
  rm(ret);

  # inputs
  i_cnt = length(ex_x[1,]);

  # hidden layer, 40% of inputs count
  h_cnt = round(i_cnt * 0.4);
  cat("hidden layer contains", h_cnt, "neurons\n");

  # neurons in the output layer
  n_cnt = length(ex_d[1,]);

  fn_k = 1;

  bh = seq(1, h_cnt);
  b = seq(1, n_cnt);

  # desired output, for learning
  d = rep(0, n_cnt);

  # weights to remember
  w_mem_step = 3;

  # initialize weights (x and h)
  wh = array(0, dim = c(w_mem_step, h_cnt, i_cnt));
  w = array(0, dim = c(w_mem_step, n_cnt, h_cnt));

  # run auto training
  i = 0;
  while (i < 20) {

    id = (i %% n_cnt) + 1;
    x = ex_x[id,];
    d = ex_d[id,];

    multi_net_calc(h_cnt, n_cnt, fn_k);
    multi_net_learn(d, h_cnt, n_cnt);

    i = i + 1;
  }

  plot_weights(w[2,,]);

  # test run
  for (i in seq(1, 5)) {

    cat("\nTest #", i, ": ");

    x = ex_data_for_test(i);

    multi_net_calc(h_cnt, n_cnt, fn_k);

    ex_analyze_results(y);
  }

}
