# Copyright (c) 2015 Alexey Kodanev <akodanev@gmail.com>
# All Rights Reserved.
#
# Multi-layer artificial neural network with learning (+1 hidden layer)
# ===================================================================

rm(list=ls());

source("activation_fn.R");
source("plot_data.R");

#source("ex_data_img16x16.R");
source("ex_data_img5x7.R");

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
  lr = 0.1;
  w0 = w;
  a = 0.0001;

  err = (t - y) * act_sigmoid_deriv_y_fn(y, k);

  bi = which.min(err);

  w[2,bi,] = w[2,bi,] + a * w[1,bi,] + lr * err[bi] * hy;

  b[2,bi] = b[2,bi] + a * b[1,bi] + lr * err[bi];

  w[1,bi,] = w[2,bi,];
  b[1,bi] = b[2,bi];
  # update weights in the hidden layer

  err_h = rep(0, h_cnt);
  for (i in seq(1, h_cnt)) {
    err_h[i] = act_sigmoid_deriv_y_fn(hy[i], k) * (err %*% w0[2,,i]);
  }

  bi_h = which.min(err_h);

  if (all(bi_h == 1)) {
    idx = c(bi_h, h_cnt, 2);
  } else if (all(bi_h == h_cnt)) {
    idx = c(bi_h, 1, h_cnt - 1);
  } else {
    idx = c(bi_h, bi_h - 1, bi_h + 1);
  }

  f = c(1, 0.3, 0.3);

  for (i in seq(1, 3)) {
    id = idx[i];
    wh[2,id,] = wh[2,id,] + a * wh[1,id,] + f[i] * lr * err_h[id] * x;
    bh[2,id] = bh[2,id] + a * bh[1,id] + f[i] * lr * err_h[id];
    wh[1,id,] = wh[2,id,];
    bh[1,id] = bh[2,id];
  }

  return(list(wh, bh, w, b));
}

multi_net_init_weights = function(y_cnt, x_cnt, rand_value)
{
  wmem = 2;
  
  w = array(0, dim = c(wmem, y_cnt, x_cnt));
  
  # Generally the more connections to a neuron that there are, the smaller the
  # initial random numbers should be. This is so that the sum of the neuron is
  # roughly within the active area of the sigmoid function (between -4 and +4).
  for (i in seq(1, y_cnt)) {
    w[1,i,] = runif(x_cnt, -rand_value, rand_value);
  }
  w[2,,] = w[1,,];
  
  # check the sum of weights in each neuron, find maximum
  w_sum = rep(0, y_cnt);
  for (i in seq(1, y_cnt)) {
    w_sum[i] = w[1,i,] %*% rep(1, x_cnt);
  }
  cat("maximum neuron random weight sum is [", max(w_sum), "]\n");
  if (abs(max(w_sum)) > 4) {
    print("WARNING: please lower random weights range");
  }

  return (w);
}

multi_net_init_bias = function(cnt, rand_value)
{
  wmem = 2;
  
  b = array(0, dim = c(wmem, cnt));
  b[1,] = runif(cnt, -rand_value, rand_value);
  b[2,] = b[1,];
  
  return (b);
}

multi_net = function()
{
  ret = ex_data_for_learn();
  ex_x = ret[[1]]; ex_t = ret[[2]];
  rm(ret);

  # inputs
  i_cnt = length(ex_x[1,]);

  # hidden layer, 40% of inputs count
  h_cnt = round(i_cnt * 0.3);
  cat("hidden layer contains", h_cnt, "neurons\n");

  # neurons in the output layer
  n_cnt = length(ex_t[1,]);
  
  # k for activation functions
  k = 1;

  # hidden layer
  wh = multi_net_init_weights(h_cnt, i_cnt, 0.4);
  bh = multi_net_init_bias(h_cnt, 0.1);
  hs = rep(0, h_cnt);
  hy = rep(0, h_cnt);
  
  # output layer
  w = multi_net_init_weights(n_cnt, h_cnt, 0.7);  
  b = multi_net_init_bias(n_cnt, 1);
  s = rep(0, n_cnt);
  y = rep(0, n_cnt);
  # training output
  t = rep(0, n_cnt);
  
  print("run auto training...");
  i = 0;
  total_error = 0;
  sum_error = c(0,0);
  tr_start = 0; tr_end = 0;
  
  while (i < 30000) {

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
    
    # calculate total error of the training cycle
    if (i != 0 && (i %% n_cnt == 0)) {          
      tr_end = i + 1;
      tr_error = total_error[seq(tr_start, tr_end)];
      
      if (tr_start != 0) {
        sum_error = c(sum_error, tr_error %*% rep(1, length(tr_error)));
        avg_error = c(avg_error, sum_error[length(sum_error)] / length(tr_error));
      } else {
        sum_error = c(tr_error %*% rep(1, length(tr_error)));
        avg_error = sum_error[length(sum_error)] / length(tr_error);
      }

      tr_start = tr_end;

      if ((i > n_cnt) && (avg_error[length(avg_error)] <= 0.25)) {
        break;
      }
    }

    i = i + 1;
  }

  # remove first big numbers
  avg_error = avg_error[30:length(avg_error)];

  # Plot data
  p = plot_set_parameters(2,2);

  plot_y(avg_error, "avg_error");
  plot_weights(wh, "hidden neuron's weights");
  plot_weights(w, "output neuron's weights");

  plot_cleanup(p);

  # test run
  for (i in seq(1, ex_alp_len)) {

    cat("\nTest #", i, ": ");

    x = ex_data_for_test(i);

    ret = multi_net_calc(x, i_cnt, hs, hy, wh, bh, h_cnt, s, w, b, y, n_cnt, k);
    hs = ret[[1]]; hy = ret[[2]]; s = ret[[3]]; y = ret[[4]];

    ex_analyze_results(y);
  }

}
