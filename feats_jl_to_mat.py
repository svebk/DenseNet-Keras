import json
import base64
import numpy as np
import scipy.io as sio
from argparse import ArgumentParser

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("-i", "--input_path", dest="input_path", default='./feats.jl')
  parser.add_argument("-o", "--output_path", dest="output_path", default='./feats.mat')
  parser.add_argument("-f", "--feat_type", dest="feat_type", default='feat_densnet121')
  opts = parser.parse_args()

  sids = []
  feats = []
  img_names = []
  with open(opts.input_path, 'rt') as inf:
    for line in inf:
      row = json.loads(line)
      img_names.append(row["img_name"])
      sids.append(float(row["class_id"]))
      feat_dtype = row[opts.feat_type+"_dtype"]
      if feat_dtype == "float32":
        dtype = np.float32
      elif feat_dtype == "float64":
        dtype = np.float64
      else:
        raise ValueError("Unknown dtype: {}".format(feat_dtype))
      feat = np.frombuffer(base64.b64decode(row[opts.feat_type]), dtype=dtype)
      feats.append(feat)

  mat_dict = dict()
  # SUBJECT_ID
  #mat_dict["SUBJECT_ID"] = np.asarray(sids)
  mat_dict["SUBJECT_ID"] = sids
  # feats
  #mat_dict["feats"] = np.asarray(feats)
  mat_dict["feats"] = feats
  mat_dict["img_names"] = img_names
  with open(opts.output_path, 'wb') as outf:
    sio.savemat(outf, mat_dict)