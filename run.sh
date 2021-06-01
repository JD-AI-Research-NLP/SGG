export CUDA_VISIBLE_DEVICES='0'

python run_summarization.py --mode=train --data_path=dataset/train.summary_*  --vocab_path=dataset/vocab --log_root=summary-hinet  --exp_name=myexperiment   


#nohup python run_summarization.py --mode=decode --data_path=dataset/test_summary_inspec_* --beam_size=4 --vocab_path=dataset/vocab --log_root=summary-hinet  --exp_name=myexperiment  --single_pass=True > inspec.124162 2>&1 &

#nohup python run_summarization.py --mode=decode --data_path=dataset/test_summary_krapivin_* --beam_size=10  --vocab_path=dataset/vocab --log_root=summary-hinet  --exp_name=myexperiment  --single_pass=True > summary1.atten 2>&1 &

#nohup python run_summarization.py --mode=decode --data_path=dataset/test_summary_krapivin_* --beam_size=10  --vocab_path=dataset/vocab --log_root=HiNet  --exp_name=myexperiment  --single_pass=True > keyphrase1.atten 2>&1 &


#nohup python run_summarization.py --mode=decode --data_path=dataset/test_summary_nus_* --beam_size=6  --vocab_path=dataset/vocab --log_root=summary-hinet  --exp_name=myexperiment  --single_pass=True > nus.124162 2>&1 &

#nohup python run_summarization.py --mode=decode --data_path=dataset/test_summary_semeval_*  --beam_size=7   --vocab_path=dataset/vocab --log_root=summary-hinet  --exp_name=myexperiment  --single_pass=True > semeval.124162 2>&1 &

#nohup python run_summarization.py --mode=decode --data_path=dataset/test_summary_kp20k_* --beam_size=8 --vocab_path=dataset/vocab --log_root=summary-hinet  --exp_name=myexperiment  --single_pass=True > kp20k.124162 2>&1 &

#nohup python run_summarization.py --mode=decode --data_path=dataset/test_summary_kp20k2_* --beam_size=108 --vocab_path=dataset/vocab --log_root=summary-hinet  --exp_name=myexperiment  --single_pass=True > kp20k2.85467 2>&1 &

#nohup python run_summarization.py --mode=decode --data_path=dataset/test_summary_kp20k3_* --beam_size=109 --vocab_path=dataset/vocab --log_root=summary-hinet  --exp_name=myexperiment  --single_pass=True > kp20k3.85467 2>&1 &

#nohup python run_summarization.py --mode=decode --data_path=dataset/test_summary_kp20k0_* --beam_size=106 --vocab_path=dataset/vocab --log_root=summary-hinet  --exp_name=myexperiment  --single_pass=True > kp20k0.85467 2>&1 &

