# Copyright (c) 2015 Alexey Kodanev <akodanev@gmail.com>
# All Rights Reserved.

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
