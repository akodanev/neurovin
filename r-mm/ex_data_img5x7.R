# Copyright (c) 2015 Alexey Kodanev <akodanev@gmail.com>
# All Rights Reserved.

# file name convention:
# ex_alp[id].png - files for training/learning;
# ex_alp[id]#.png - files for testing the algorithms;

# outputs connected to the following object
ex_alp = c('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z');

ex_alp_len = 16; # length(ex_alp);

ex_analyze_results = function(y)
{
  id = -1;
  uncertanty = -1;

  for (i in seq(1, length(y))) {
    if (y[i] >= 0.9) {
      id = i;
      uncertanty = uncertanty + 1;
    }
  }

  if (id >= 0 && uncertanty == 0) {
      cat(">OK, I know it, this is letter", ex_alp[id], "\n");
      return (0);
  }

  cat(">Hmm, not exactly sure what it is", round(y, 5), "\n");
}

ex_data_for_learn = function()
{
  library(png);

  xv = rep(0, 35);
  ex_x = array(0, dim = c(ex_alp_len, 35));
  ex_d = diag(1, ex_alp_len);

  for (i in seq(1, ex_alp_len)) {
    fname = sprintf("img5x7/%s.png", ex_alp[i]);
    img = readPNG(fname);
    ex_x[i,] = as.vector(t(img[,,4]));
    # tmp_x = t(img[,,4]);
    # ex_x[i,] = alg_matrix_to_spiral(xv, tmp_x, 1, 16, 1, 16);
  }

  return (list(ex_x, ex_d));
}

ex_data_for_test = function(i, k = 0)
{
  library(png);

  xv = rep(0, 35);

  fname = sprintf("img5x7/%s.png", ex_alp[i]);
  #fname = sprintf("img16x16/%s.png", ex_alp[i]);

  img = readPNG(fname);
  # tmp_x = t(img[,,4]);
  # ex_x[i,] = alg_matrix_to_spiral(xv, tmp_x, 1, 16, 1, 16);
  xv = as.vector(t(img[,,4]))

  cat("read file", fname, "\n");

  return (xv);
}
