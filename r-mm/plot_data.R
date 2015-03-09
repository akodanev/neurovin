# Copyright (c) 2015 Alexey Kodanev <akodanev@gmail.com>
# All Rights Reserved.
#

plot_set_parameters = function(cnt,n)
{
  plot_count = cnt;
  #bottom, left, top, right
  op<-par(mfrow=c(plot_count,n), mar=c(2,5,1,1), xaxs='i',lab=c(12, 5, 7));  
}

plot_cleanup = function()
{
  par(op); # - reset to default
}

plot_weights = function(w, str)
{
  x = seq(1, length(w[1,1,]));
  plot_xy2(x, min(x), max(x), "x", w[1,,], str);
}

plot_functions = function()
{
  rm(list=ls());

  x = seq(-10, 10, 0.5);

  y = act_sigmoid_fn(x, 0.8);
  plot_xy(x, min(x), max(x), "x", y, min(y), max(y), "y");
}

plot_y = function(y, str)
{
  x = seq(1, length(y));
  plot_xy(x, 1, max(x), "steps", y, min(y), max(y), str);
}

plot_xy = function(x, min_x, max_x, x_str, y, min_y, max_y, y_str)
{
  col_1="#0000FF";
  col_2="#FF0000";
  col_grid = "#5F5F5F";

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

  points(x, y, type="l", pch = 21, col = col_2, lwd = 2, lty = 2);
}

plot_xy2 = function(x, min_x, max_x, x_str, y, y_str)
{
  
  require(RColorBrewer);
  
  col_grid = "#5F5F5F";

  # setting up coord. sys.
  l = length(y[1,]);
  
  cols = brewer.pal(length(y[,1]), "Set3");
  
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
}
