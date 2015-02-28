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
    hs[i] = bh[2,i] + wh[2,i,] %*% x;    
  }
  
  hy = act_sigmoid_fn(hs, k);
  
  for (i in seq(1, n_cnt)) {
    s[i] = b[2,i] + w[2,i,] %*% hy;
  }    

  y = act_sigmoid_fn(s, k);
  
  return(list(hs, hy, s, y));
}

multi_net_learn = function(x, i_cnt, hs, wh, bh, hy, h_cnt, s, w, b, y, t, n_cnt, k)
{
  
  # update weights in the output layer
  lr = 2;
  w0 = w;
  a = 0.01;
  err = (t - y) * act_sigmoid_deriv_y_fn(y, k);
  for (i in seq(1, n_cnt)) {
    w[2,i,] = w[2,i,] + a * w[1,i,] + lr * err[i] * hy;    
  }
  b[2,] = b[2,] + a * b[1,] + lr * err;
  
  w[1,,] = w[2,,];
  b[1,] = b[2,];
  # update weights in the hidden layer
  
  err_h = rep(0, h_cnt);
  for (i in seq(1, h_cnt)) {
    err_h[i] = act_sigmoid_deriv_y_fn(hy[i], k) * (err %*% w0[2,,i]);
  }
  
  for (i in seq(1, h_cnt)) {
    wh[2,i,] = wh[2,i,] + a * wh[1,i,] + lr * err_h[i] * x;      
  }
  bh[2,] = bh[2,] + a * bh[1,] + lr * err_h;
    
  wh[1,,] = wh[2,,];
  bh[1,] = bh[2,];
  
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
  h_cnt = 12; #round(i_cnt * 0.2);
  cat("hidden layer contains", h_cnt, "neurons\n");

  # neurons in the output layer
  n_cnt = length(ex_t[1,]);
  
  # k for activation functions
  k = 1;

  # weight memory
  wmem = 2;
  
  # hidden layer
  wh = array(0, dim = c(wmem, h_cnt, i_cnt));
  for (i in seq(1, h_cnt)) {
    wh[1,i,] = runif(i_cnt, -1, 1);
  }
  wh[2,,] = wh[1,,];
  
  bh = array(0, dim = c(wmem, h_cnt));
  bh[1,] = runif(h_cnt, -1, 1);
  bh[2,] = bh[1,];
  hs = rep(0, h_cnt);
  hy = rep(0, h_cnt);
  
  # output layer
  w  = array(0, dim = c(wmem, n_cnt, h_cnt));
  for (i in seq(1, n_cnt)) {
    w[1,i,] = runif(h_cnt, -1, 1);
  }
  w[2,,] = w[1,,];

  b = array(0, dim = c(wmem, n_cnt));
  b[1,] = runif(n_cnt, -1, 1);
  b[2,] = b[1,];
  s = rep(0, n_cnt);
  y = rep(0, n_cnt);
  
  t = rep(0, n_cnt);
  
  print("run auto training...");

  i = 0;
  total_error = 0;
  
  while (i < 150) {

    id = (i %% n_cnt) + 1;
    x = ex_x[id,];
    t = ex_t[id,];
 
    ret = multi_net_calc(x, i_cnt, hs, hy, wh, bh, h_cnt, s, w, b, y, n_cnt, k);
    hs = ret[[1]]; hy = ret[[2]]; s = ret[[3]]; y = ret[[4]];

    ret = multi_net_learn(x, i_cnt, hs, wh, bh, hy, h_cnt, s, w, b, y, t, n_cnt, k);
    wh = ret[[1]]; bh = ret[[2]]; w = ret[[3]]; b = ret[[4]];
    
    if (i != 0) {
      total_error = c(total_error, abs(t - y) %*% rep(1, length(y)));        
    } else {
      total_error = c(abs(t - y) %*% rep(1, length(y)));
    }
    
    i = i + 1;
  }
  
  plot_y(total_error, "total_error");
  
  #plot_weights(wh);

  # return(0);
  # test run
  for (i in seq(1, 6)) {

    cat("\nTest #", i, ": ");

    x = ex_data_for_test(i);

    ret = multi_net_calc(x, i_cnt, hs, hy, wh, bh, h_cnt, s, w, b, y, n_cnt, k);
    hs = ret[[1]]; hy = ret[[2]]; s = ret[[3]]; y = ret[[4]];

    ex_analyze_results(y);
  }

}
