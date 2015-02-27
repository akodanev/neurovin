# Copyright (c) 2015 Alexey Kodanev <akodanev@gmail.com>
# All Rights Reserved.
#
# Multi-layer artificial neural network with learning (+1 hidden layer)
# ===================================================================

source("activation_fn.R");
source("plot_data.R");

source("ex_data_img16x16.R");

multi_net_calc = function(x, i_cnt, hs, hy, wh, bh, h_cnt, s, w, b, y, n_cnt, k)
{
  for (i in seq(1, h_cnt)) {
    hs[i] = bh[i] + wh[i,] %*% x;    
  }
  
  hy = act_sigmoid_fn(hs, k);
  
  for (i in seq(1, n_cnt)) {
    s[i] = b[i] + w[i,] %*% hy;
  }    

  y = act_sigmoid_fn(s, k);
  
  return(list(hs, hy, s, y));
}

multi_net_learn = function(x, i_cnt, hs, wh, bh, hy, h_cnt, s, w, b, y, t, n_cnt, k)
{
  
  # update weights in the output layer
  d = act_sigmoid_deriv_y_fn(y, k) * (t - y);
  
  for (i in seq(1, n_cnt)) {
    w[i,] = w[i,] + d[i] * hy;    
  }
  b = b + d;
  
  # update weights in the hidden layer
  ds = rep(0, h_cnt);
  for (i in seq(1, h_cnt)) {
    ds[i] = d %*% w[,i];
  }
  
  dh = act_sigmoid_deriv_y_fn(hy, k) * ds;
  
  for (i in seq(1, h_cnt)) {
    wh[i,] = wh[i,] + dh[i] * x;      
  }
  bh = bh + dh;
 
  return(list(wh, bh, w, b));
}

multi_net = function()
{
  ret = ex_data_for_learn();
  ex_x = ret[[1]]; ex_t = ret[[2]];
  rm(ret);

  # inputs
  i_cnt = length(ex_x[1,]);

  # hidden layer, 40% of inputs count
  h_cnt = round(i_cnt * 0.4);
  cat("hidden layer contains", h_cnt, "neurons\n");

  # neurons in the output layer
  n_cnt = length(ex_t[1,]);

  # k for activation functions
  k = 1;

  # hidden layer
  wh = array(0, dim = c(h_cnt, i_cnt));
  bh = rep(0, h_cnt);
  hs = rep(0, h_cnt);
  hy = rep(0, h_cnt);
  
  # output layer
  w  = array(0, dim = c(n_cnt, h_cnt));
  b = rep(0, n_cnt);
  s = rep(0, n_cnt);
  y = rep(0, n_cnt);
  
  d = rep(0, n_cnt);

  print("run auto training...");

  i = 0;
  while (i < 10) {

    id = (i %% n_cnt) + 1;
    x = ex_x[id,];
    t = ex_t[id,];

    ret = multi_net_calc(x, i_cnt, hs, hy, wh, bh, h_cnt, s, w, b, y, n_cnt, k);
    hs = ret[[1]]; hy = ret[[2]]; s = ret[[3]]; y = ret[[4]];

    ret = multi_net_learn(x, i_cnt, hs, wh, bh, hy, h_cnt, s, w, b, y, t, n_cnt, k);
    wh = ret[[1]]; bh = ret[[2]]; w = ret[[3]]; b = ret[[4]];

    i = i + 1;
  }
  
  plot_weights(w);

  # test run
  for (i in seq(1, 5)) {

    cat("\nTest #", i, ": ");

    x = ex_data_for_test(i);

    ret = multi_net_calc(x, i_cnt, hs, hy, wh, bh, h_cnt, s, w, b, y, n_cnt, k);
    hs = ret[[1]]; hy = ret[[2]]; s = ret[[3]]; y = ret[[4]];

    ex_analyze_results(y);
  }

}
