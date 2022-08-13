# Heterogeneous Graph Network-based Software Developer Recommendation
## Requirements
pip install  tensorboardX gensim==4.0.1 nltk \
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu111.html \
import nltk \
nltk.download('stopwords') \
nltk.download('punkt')

### Running the code

'''
python HGDR_solver.py --dataset=Tensorflow --num_core=10 --num_feat_core=10 --sampling_strategy=random  --emb_dim=64 --repr_dim=16  --hidden_size=64 --meta_path_steps=2,2,2,2,2  --init_eval=true --gpu_idx=0 --runs=2 --epochs=30 --batch_size=1024 --save_every_epoch=10 --metapath_test=true --head_tail_test=false --ssl=true --feat_drop=0.2 --tau=0.5  --dropout=0.2  --ssl_coff=0.2 --side_info=true --side_info_vector_size=64 --side_info_window=8 --side_info_min_count=3 --side_info_epoch=8
'''
  

  
