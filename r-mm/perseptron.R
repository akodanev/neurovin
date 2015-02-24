
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

# Learning Algorithm
# ========

# 1. Back-propagation learning
# ----------------------------
ln_back_prop_fn = function()
{
  ln_speed = 0.1;
  ln_moment = 0.1;
  
  in_cnt = 2; 
  neurons = c(2, 1);
  layers_cnt = length(neurons);
  
  n_err = matrix(0, nrow = layers_cnt, ncol = max(neurons));
  print(n_err);
  w = array(0, dim = c(layers_cnt, max(neurons), in_cnt));
  print(w);
}

# 2. Back-propagation learning - 1 layer
# ----------------------------
ln_back_prop_fn2 = function()
{
  ex_x = array(0, dim = c(3,4));
  ex_d = array(0, dim = c(3,3));  
    
  #slash
  ex_x[1,1] = 0; ex_x[1,2] = 1; ex_x[1,3] = 1; ex_x[1,4] = 0;
  ex_d[1,1] = 1; ex_d[1,2] = 0; ex_d[1,3] = 0;
  
  # backslash
  ex_x[2,1] = 1; ex_x[2,2] = 0; ex_x[2,3] = 0; ex_x[2,4] = 1;
  ex_d[2,1] = 0; ex_d[2,2] = 1; ex_d[2,3] = 0;
  
  # ar
  ex_x[3,1] = 1; ex_x[3,2] = 1; ex_x[3,3] = 0; ex_x[3,4] = 1;
  ex_d[3,1] = 0; ex_d[3,2] = 0; ex_d[3,3] = 1;
  
  x = ex_x[1,];
  
  i_cnt = length(x);
  print("x = "); print(x);
  
  n_cnt = c(3);
  
  d = ex_d[1,];

  s = rep(0, n_cnt);
  
  b = rep(0, n_cnt); print("b = "); print(b);
  
  mem_step = 3;
  
  w = array(1, dim = c(mem_step, n_cnt, i_cnt));
  
  print("w[3,] = "); print(w[3,,]);
  
  learn = 1;
  steps = 1;
  while (1) {
    steps = steps + 1;
    
    for (i in seq(1, n_cnt)) {
      s[i] = b[i] + w[2,i, ] %*% x;
    }
  
    y = act_sigmoid_fn(s);
            
    if (steps %% 3 == 0) {
      print("----------------");
      print("x = "); print(x);
      print("y = "); print(y);

      res = readline('Continue or run example (y, n, num(1..3), p(plot weights))?');
      learn = 1;
      if (res == "n") {
        break;
      } else if (res == "1") {
        x = ex_x[1,];
        d = ex_d[1,];
      } else if (res == "2") {
        x = ex_x[2,];
        d = ex_d[2,];
      } else if (res == "3") {
        x = ex_x[3,];
        d = ex_d[3,];
      } else if (res == "p") {
        plot_weights(w[3,,]);
      } else {
        learn = 0;
        x = c(1, 1, 1, 1);
      }
    }

    if (!learn)
      next;
    # Learning...
    # adjust weights if needed
    n_err = (d - y); 
    #print("n_err = "); print(n_err);

    a = 0.01;
    nu = 0.45;

    for (i in seq(1, n_cnt)) {
      w[3,i,] = w[2,i,] + a * w[1,i,] + (nu * n_err[i]) * x;
    }

    w[1,,] = w[2,,];
    w[2,,] = w[3,,];

    # new weights
    print("w[3,i] = "); print(w[3,,]);
  }

}

plot_weights = function(w)
{
  x = seq(1, length(w[1,]));
  plot_xy2(x, min(x), max(x), "x", w, "weights");
}

multilayer_perceptron = function()
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
  print(min(y));print(max(y));
  plot(seq(min_x, max_x, step_x), seq(min(y), max(y), step),
    type ='n', ylab=y_str);
  grid(lwd = 1, col = col_grid);

  #-------------------------------------------------
  for (i in seq(1, length(y[,1]))) {
     points(x, y[i,], type="p", col = cols[i]);
  }

  par(op); # - reset to default
}

