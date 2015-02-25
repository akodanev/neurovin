
# Activation functions
# ====================

# 1. sigmoid function, where y in (0..1) range
# --------------------------------------------
act_sigmoid_fn = function(x, k = 1)
{
  return (1 / (1 + exp(-x * k)));
}
act_sigmoid_deriv_fn = function(x, k = 1)
{
  y = act_sigmoid_fn(x, k);
  return (k * y * (1 - y));
}

# 2. bipolar sigmoid function, y in (-1..1) range
# -----------------------------------------------
act_bsigmoid_fn = function(x, k = 1)
{
  return (2 / (1 + exp(-x * k)) - 1);
}
act_bsigmoid_deriv_fn = function(x, k = 1)
{
  y = act_bsigmoid_fn(x, k);
  return (k * (1 - y * y) / 2);
}

pos = function(x)
{
  if (x >= 0)
    return (1);
  
  return (0);
}
# 3. Threshold function, y is 0 or 1
act_threshold_fn = function(x, k = 1)
{
  y = x;
  for (i in seq(1, length(x))) {
    y[i] = pos(x[i]);
  }
  return (y);
}
act_threshold_deriv_fn = function(x, k = 1)
{
  return (0);
}


# 1. Single-layer artificial neural network with learning
# -------------------------------------------------------

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
  ret = ex_data_pic16x16_for_learn();
  #ret = ex_data_x4d3();
  ex_x = ret[[1]]; ex_d = ret[[2]];
  rm(ret);
  
  # inputs
  i_cnt = length(ex_x[1,]);
  
  # neurons in the layer
  n_cnt = 3;
  
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
    
    x = ex_data_pic16x16_for_test(i);
    
    y = single_net_calc(b, w, x, n_cnt, fn_k);
    
    analyze_results(y);
  }
  
}

analyze_results = function(y)
{
  alp = c('A', 'B', 'C');
  id = -1;
  uncertanty = -1;
  
  for (i in seq(1, length(y))) {
    if (y[i] >= 0.9) {
      id = i;
      uncertanty = uncertanty + 1;
    }
  }
  
  if (id >= 0 && uncertanty == 0) {
      cat(">OK, I know it, this is letter", alp[id], "\n");
      return (0);
  }
  
  cat(">Hmm, not exactly sure what it is", round(y, 5), "\n");
}

ex_data_pic16x16_for_learn = function()
{
  library(png);

  xv = rep(0, 256);
  ex_x = array(0, dim = c(3,256));
  ex_d = array(0, dim = c(3,3));

  id = c("img16x16/A.png", "img16x16/B.png", "img16x16/C.png");

  for (i in seq(1, 3)) {
    img = readPNG(id[i]);    
    #tmp_x = t(img[,,4]);
    #ex_x[i,] = alg_matrix_to_spiral(xv, tmp_x, 1, 16, 1, 16);
    ex_x[i,] = as.vector(t(img[,,4]));        
  }

  ex_d[1,1] = 1; ex_d[1,2] = 0; ex_d[1,3] = 0; # A
  ex_d[2,1] = 0; ex_d[2,2] = 1; ex_d[2,3] = 0; # B
  ex_d[3,1] = 0; ex_d[3,2] = 0; ex_d[3,3] = 1; # C

  return (list(ex_x, ex_d));
}

ex_data_pic16x16_for_test = function(i)
{
  library(png);
  
  xv = rep(0, 256);
  
  id = c("img16x16/A0.png", "img16x16/B0.png", "img16x16/C0.png", "img16x16/D0.png",
         "img16x16/E0.png");

  img = readPNG(id[i]);
    #tmp_x = t(img[,,4]);
    #ex_x[i,] = alg_matrix_to_spiral(xv, tmp_x, 1, 16, 1, 16);
  xv = as.vector(t(img[,,4]))

  cat("read file", id[i], "\n");
  
  return (xv);
}

plot_weights = function(w)
{
  x = seq(1, length(w[1,]));
  plot_xy2(x, min(x), max(x), "x", w, "weights");
}

plot_functions = function()
{
  rm(list=ls());
  
  x = seq(-10, 10, 0.5);

  y = act_sigmoid_deriv_fn(x, 0.8);
  plot_xy(x, min(x), max(x), "x", y, min(y), max(y), "y");
}

plot_xy = function(x, min_x, max_x, x_str, y, min_y, max_y, y_str)
{
  col_1="#0000FF";
  col_2="#FF0000";
  col_grid = "#5F5F5F";

  plot_count = 1;
  #bottom, left, top, right
  op<-par(mfrow=c(plot_count,1), mar=c(2,5,1,1), xaxs='i',lab=c(12, 5, 7));
  
  print('plot name: ');
    
  # setting up coord. sys.
  l = length(y);
  if (l == 1) {
    step = 0;
  } else {
    step = abs(max_y - min_y) / (l - 1);
  }

  plot(x, seq(min_y, max_y, step), type ='n', ylab=y_str);
  grid(lwd = 1, col = col_grid);
  
  #-------------------------------------------------
  points(x, y, type="p", col = col_1);
  # points(v_t_faults,v_mean, type="p",col="#00FF00", pch = 10);
  
  par(op);#- reset to default
}

plot_xy2 = function(x, min_x, max_x, x_str, y, y_str)
{
  cols = c("red", "blue", "black", "green");
  col_grid = "#5F5F5F";

  plot_count = 1;
  #bottom, left, top, right
  op<-par(mfrow=c(plot_count,1), mar=c(2,5,1,1), xaxs='i',lab=c(12, 5, 7));

  # setting up coord. sys.
  l = length(y[1,]);
  max_x = max(x) + 1;
  min_x = min(x) - 1;
  if (l == 1) {
    step = 0;
  } else {
    step = abs(max(y) - min(y)) / (l - 1);
    step_x = abs(max_x - min_x) / (l - 1);
  }
  
  plot(seq(min_x, max_x, step_x), seq(min(y), max(y), step),
    type ='n', ylab=y_str);
  grid(lwd = 1, col = col_grid);

  #-------------------------------------------------
  for (i in seq(1, length(y[,1]))) {
     points(x, y[i,], type="p", pch = 21, bg = cols[i], cex = 1.5);
  }

  par(op); # - reset to default
}

# Other algorithms
# ==================

alg_matrix_to_spiral = function(x, m, rs = 0, re = 0, cs = 0, ce = 0, id = 1,
                                rev = 0, row = 0)
{
  if (rs > re && cs > ce) {
    return (x);
  }
  
  if (row == 0) {
    if (rev == 0) {
      for (i in seq(cs, ce)) {
        x[id] = m[rs,i];
        id = id + 1;
      }
      if (id > length(x)) {
        return (x);
      }
    }
    rs = rs + 1;
    row = 1;
    rev = 0;
  }
  
  if (row == 1) {
    if (rev == 0) {
      for (i in seq(rs, re)) {
        x[id] = m[i,ce];
        id = id + 1;
      }
      if (id > length(x)) {
        return (x);
      }
    }
    ce = ce - 1;
    row = 0;
    rev = 1;
  }
  
  if (row == 0) {
    if (rev == 1) {
      for (i in seq(ce, cs)) {
        x[id] = m[re,i];
        id = id + 1;
      }
      if (id > length(x)) {
        return (x);
      }
    }
    re = re - 1;
    row = 1;
    rev = 1;
  }
  
  if (row == 1) {
    if (rev == 1) {
      for (i in seq(re, rs)) {
        x[id] = m[i,cs];
        id = id + 1;
      }
      if (id > length(x)) {
        return (x);
      }
    }
    cs = cs + 1;
    row = 0;
    rev = 0;
  }
  
  x = alg_matrix_to_spiral(x, m, rs, re, cs, ce, id, rev, row);
  
  return (x);
}



# Functions that are no longer used (will be removed)
# ===================================================

# ex_data_x4d3 = function()
# {
#   ex_x = array(0, dim = c(3,4));
#   ex_d = array(0, dim = c(3,3));  
#   
#   #slash
#   ex_x[1,1] = 0; ex_x[1,2] = 1; ex_x[1,3] = 1; ex_x[1,4] = 0;
#   ex_d[1,1] = 1; ex_d[1,2] = 0; ex_d[1,3] = 0;
#   
#   # backslash
#   ex_x[2,1] = 1; ex_x[2,2] = 0; ex_x[2,3] = 0; ex_x[2,4] = 1;
#   ex_d[2,1] = 0; ex_d[2,2] = 1; ex_d[2,3] = 0;
#   
#   # ar
#   ex_x[3,1] = 1; ex_x[3,2] = 1; ex_x[3,3] = 0; ex_x[3,4] = 1;
#   ex_d[3,1] = 0; ex_d[3,2] = 0; ex_d[3,3] = 1;
#   
#   return (list(ex_x, ex_d));
# }

# ex_data_x4 = function()
# {
#   ex_x = array(0, dim = c(5,4));
#   ex_x[1,1] = 0; ex_x[1,2] = 1; ex_x[1,3] = 1; ex_x[1,4] = 0;
#   ex_x[2,1] = 1; ex_x[2,2] = 0; ex_x[2,3] = 0; ex_x[2,4] = 1;
#   ex_x[3,1] = 1; ex_x[3,2] = 1; ex_x[3,3] = 0; ex_x[3,4] = 1;
#   ex_x[4,1] = 1; ex_x[4,2] = 1; ex_x[4,3] = 1; ex_x[4,4] = 1;
#   ex_x[5,1] = 0; ex_x[5,2] = 1; ex_x[5,3] = 1; ex_x[5,4] = 1;
#   
#   return (ex_x);
# }

#check_point = function()
#{        
  #   if (steps %% 3 != 0)
  #     return;
  #   
  #   print("----------------");
  #   print("x = "); print(x);
  #   print("y = "); print(y);
  # 
  #   res = readline('Continue or run example (y, n, num(1..3), p(plot weights))?');
  #   learn = 1;
  #   if (res == "n") {
  #     return;
  #   } else if (res == "1") {
  #     x = ex_x[1,];
  #     d = ex_d[1,];
  #   } else if (res == "2") {
  #     x = ex_x[2,];
  #     d = ex_d[2,];
  #   } else if (res == "3") {
  #     x = ex_x[3,];
  #     d = ex_d[3,];
  #   } else if (res == "p") {
  #     plot_weights(w[3,,]);
  #   } else {
  #     learn = 0;
  #     x = c(1, 1, 1, 1);
  #   }
#}
