python3 train.py --train run_1.train --devel run_1.dev --test run_1.test --word-emb ../w2v/s24_wf_200_skgram.bin --lemma-emb ../w2v/s24_lemma_200_skgram.bin --maxrank-emb 400000 --word-seq-len 600 --model-file model.run_1 --classname DocuClassifierMultioutCNN