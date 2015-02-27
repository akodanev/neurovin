# Copyright (c) 2015 Alexey Kodanev <akodanev@gmail.com>
# All Rights Reserved.

ex_analyze_results = function(y)
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

ex_data_for_learn = function()
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

ex_data_for_test = function(i)
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
