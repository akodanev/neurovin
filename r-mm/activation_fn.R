# Copyright (c) 2015 Alexey Kodanev <akodanev@gmail.com>
# All Rights Reserved.
#
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
