CUDA_LAUNCH_BLOCKING=1 
#python main.py --train Movie --save_dir ./data/Movie
#python main.py --load Movie --save_dir ./data/Movie --load_file ./data/Movie/model/2_512_64/16_aspect_model.tar
#python main.py --test Movie --save_dir ./data/Movie \
#               --aspect_model /home/kun_zhou/lijunyi/graph/topic/data/Movie/model/2_512_64/25_aspect_model.tar

#python main.py --train Electronic --save_dir ./data/Electronic
#python main.py --load Electronic --save_dir ./data/Electronic --load_file ./data/Electronic/model/2_512_128/12_aspect_model.tar
#python main.py --test Electronic --save_dir ./data/Electronic \
#               --aspect_model /home/kun_zhou/lijunyi/graph/topic/data/Electronic/model/2_512_128/20_aspect_model.tar

python main.py --train Book --save_dir ./data/Book
#python main.py --load Book --save_dir ./data/Book --load_file ./data/Book/model/2_512_128/18_aspect_model.tar
#python main.py --test Book --save_dir ./data/Book \
#               --aspect_model /home/kun_zhou/lijunyi/graph/topic/data/Book/model/2_512_128/29_aspect_model.tar
