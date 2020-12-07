import glob
import os.path
import random
from multiprocessing import Process

import Utils

import numpy as np
from lxml import etree
import librosa
import soundfile
import os
import tensorflow as tf
import musdb
import h5py
from itertools import chain
from matplotlib import pyplot as plt
from scipy.signal import resample

# Define songs/stems to be included as part of validation set
VAL_SET = [
''
]

def take_random_snippets(sample, keys, input_shape, num_samples):
	# Take a sample (collection of audio files) and extract snippets from it at a number of random positions
	start_pos = tf.random_uniform([num_samples], 0, maxval=sample["length"] - input_shape[0], dtype=tf.int64)
	return take_snippets_at_pos(sample, keys, start_pos, input_shape, num_samples)

def take_all_snippets(sample, keys, input_shape, output_shape):
	# Take a sample and extract snippets from the audio signals, using a hop size equal to the output size of the network
	start_pos = tf.range(0, sample["length"] - input_shape[0], delta=output_shape[0], dtype=tf.int64)
	num_samples = start_pos.shape[0]
	return take_snippets_at_pos(sample, keys, start_pos, input_shape, num_samples)

def take_snippets_at_pos(sample, keys, start_pos, input_shape, num_samples):
	# Take a sample and extract snippets from the audio signals at the given start positions with the given number of samples width
	batch = dict()
	for key in keys:
		batch[key] = tf.map_fn(lambda pos: sample[key][pos:pos + input_shape[0], :], start_pos, dtype=tf.float32)
		batch[key].set_shape([num_samples, input_shape[0], input_shape[1]])

	return tf.data.Dataset.from_tensor_slices(batch)

def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_records(sample_list, model_config, input_shape, output_shape, records_path):
	# Writes samples in the given list as TFrecords into a given path, using the current model config and in/output shapes

	# Compute padding
	if (input_shape[1] - output_shape[1]) % 2 != 0:
		print("WARNING: Required number of padding of " + str(input_shape[1] - output_shape[1]) + " is uneven!")
	pad_frames = (input_shape[1] - output_shape[1]) // 2

	# Set up writers
	num_writers = 1
	writers = [tf.python_io.TFRecordWriter(records_path + str(i) + ".tfrecords") for i in range(num_writers)]

	# Go through songs and write them to TFRecords
	all_keys = model_config["source_names"] + ["mix"]
	for sample in sample_list:
		print("Reading song")
		try:
			audio_tracks = dict()

			for key in all_keys:
				audio, _ = Utils.load(sample[key], sr=model_config["expected_sr"], mono=model_config["mono_downmix"])

				if not model_config["mono_downmix"] and audio.shape[1] == 1:
					print("WARNING: Had to duplicate mono track to generate stereo")
					audio = np.tile(audio, [1, 2])

				audio_tracks[key] = audio
		except Exception as e:
			print(e)
			print("ERROR occurred during loading file " + str(sample) + ". Skipping")
			continue

		# Pad at beginning and end with zeros
		audio_tracks = {key : np.pad(audio_tracks[key], [(pad_frames, pad_frames), (0, 0)], mode="constant", constant_values=0.0) for key in list(audio_tracks.keys())}

		# All audio tracks must be exactly same length and channels
		length = audio_tracks["mix"].shape[0]
		channels = audio_tracks["mix"].shape[1]
		for audio in list(audio_tracks.values()):
			assert(audio.shape[0] == length)
			assert (audio.shape[1] == channels)

		# Write to TFrecords the flattened version
		feature = {key: _floats_feature(audio_tracks[key]) for key in all_keys}
		feature["length"] = _int64_feature(length)
		feature["channels"] = _int64_feature(channels)
		sample = tf.train.Example(features=tf.train.Features(feature=feature))
		writers[np.random.randint(0, num_writers)].write(sample.SerializeToString())

	for writer in writers:
		writer.close()

def parse_record(example_proto, source_names, shape):
	# Parse record from TFRecord file

	all_names = source_names + ["mix"]

	features = {key : tf.FixedLenSequenceFeature([], allow_missing=True, dtype=tf.float32) for key in all_names}
	features["length"] = tf.FixedLenFeature([], tf.int64)
	features["channels"] = tf.FixedLenFeature([], tf.int64)

	parsed_features = tf.parse_single_example(example_proto, features)

	# Reshape
	length = tf.cast(parsed_features["length"], tf.int64)
	channels = tf.constant(shape[-1], tf.int64) #tf.cast(parsed_features["channels"], tf.int64)
	sample = dict()
	for key in all_names:
		sample[key] = tf.reshape(parsed_features[key], tf.stack([length, channels]))
	sample["length"] = length
	sample["channels"] = channels

	return sample

def get_dataset(model_config, input_shape, output_shape, partition):
	'''
	For a model configuration and input/output shapes of the network, get the corresponding dataset for a given partition
	:param model_config: Model config
	:param input_shape: Input shape of network
	:param output_shape: Output shape of network
	:param partition: "train", "valid", or "test" partition
	:return: Tensorflow dataset object
	'''


	# Check if pre-processed dataset is already available for this model config and partition
	dataset_name = "task_" + model_config["task"] + "_" + \
				   "sr_" + str(model_config["expected_sr"]) + "_" + \
				   "mono_" + str(model_config["mono_downmix"])
	main_folder = os.path.join(model_config["data_path"], dataset_name)

	if not os.path.exists(main_folder):

		# We have to prepare the MUSDB dataset
		print("Preparing MUSDB dataset! This could take a while...")
		dsd_train, dsd_test = getMUSDB(model_config["musdb_path"])  # List of (mix, acc, bass, drums, other, vocal) tuples

		# Pick 25 random songs for validation from MUSDB train set (this is always the same selection each time since we fix the random seed!)
		val_idx = np.random.choice(len(dsd_train), size=25, replace=False)
		train_idx = [i for i in range(len(dsd_train)) if i not in val_idx]
		print("Validation with MUSDB training songs no. " + str(val_idx))

		# Draw randomly from datasets
		dataset = dict()
		dataset["train"] = [dsd_train[i] for i in train_idx]
		dataset["valid"] = [dsd_train[i] for i in val_idx]
		dataset["test"] = dsd_test

		# MUSDB base dataset loaded now, now create task-specific dataset based on that
		if model_config["task"] == "voice":
			# Prepare CCMixter
			print("Preparing CCMixter dataset!")
			ccm = getCCMixter("CCMixter.xml")
			dataset["train"].extend(ccm)


		# Convert audio files into TFRecords now

		# The dataset structure is a dictionary with "train", "valid", "test" keys, whose entries are lists, where each element represents a song.
		# Each song is represented as a dictionary containing elements mix, acc, vocal or mix, bass, drums, other, vocal depending on the task.

		num_cores = 8

		for curr_partition in ["train", "valid", "test"]:
			print("Writing " + curr_partition + " partition...")

			# Shuffle sample order
			sample_list = dataset[curr_partition]
			random.shuffle(sample_list)

			# Create folder
			partition_folder = os.path.join(main_folder, curr_partition)
			os.makedirs(partition_folder)

			part_entries = int(np.ceil(float(len(sample_list) / float(num_cores))))
			processes = list()
			for core in range(num_cores):
				train_filename = os.path.join(partition_folder, str(core) + "_")  # address to save the TFRecords file
				sample_list_subset = sample_list[core * part_entries:min((core + 1) * part_entries, len(sample_list))]
				proc = Process(target=write_records,
							   args=(sample_list_subset, model_config, input_shape, output_shape, train_filename))
				proc.start()
				processes.append(proc)
			for p in processes:
				p.join()

	print("Dataset ready!")
	# Finally, load TFRecords dataset based on the desired partition
	dataset_folder = os.path.join(main_folder, partition)
	records_files = glob.glob(os.path.join(dataset_folder, "*.tfrecords"))
	random.shuffle(records_files)
	dataset = tf.data.TFRecordDataset(records_files)
	dataset = dataset.map(lambda x : parse_record(x, model_config["source_names"], input_shape[1:]), num_parallel_calls=model_config["num_workers"])
	dataset = dataset.prefetch(10)

	# Take random samples from each song
	if partition == "train":
		dataset = dataset.flat_map(lambda x : take_random_snippets(x, model_config["source_names"] + ["mix"], input_shape[1:], model_config["num_snippets_per_track"]))
	else:
		dataset = dataset.flat_map(lambda x : take_all_snippets(x, model_config["source_names"] + ["mix"], input_shape[1:], output_shape[1:]))
	dataset = dataset.prefetch(100)

	if partition == "train" and model_config["augmentation"]: # If its the train partition, activate data augmentation if desired
			dataset = dataset.map(Utils.random_amplify, num_parallel_calls=model_config["num_workers"]).prefetch(100)

	# Cut source outputs to centre part
	dataset = dataset.map(lambda x : Utils.crop_sample(x, (input_shape[1] - output_shape[1])//2)).prefetch(100)

	if partition == "train": # Repeat endlessly and shuffle when training
		dataset = dataset.repeat()
		dataset = dataset.shuffle(buffer_size=model_config["cache_size"])

	dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(model_config["batch_size"]))
	dataset = dataset.prefetch(1)

	return dataset

def get_path(db_path, instrument_node):
	return db_path + os.path.sep + instrument_node.xpath("./relativeFilepath")[0].text

def getMUSDB(database_path):
	mus = musdb.DB(root_dir=database_path, is_wav=False)

	subsets = list()

	for subset in ["train", "test"]:
		tracks = mus.load_mus_tracks(subset)
		samples = list()

		# Go through tracks
		for track in tracks:
			# Skip track if mixture is already written, assuming this track is done already
			track_path = track.path[:-4]
			mix_path = track_path + "_mix.wav"
			acc_path = track_path + "_accompaniment.wav"
			if os.path.exists(mix_path):
				print("WARNING: Skipping track " + mix_path + " since it exists already")

				# Add paths and then skip
				paths = {"mix" : mix_path, "accompaniment" : acc_path}
				paths.update({key : track_path + "_" + key + ".wav" for key in ["bass", "drums", "other", "vocals"]})

				samples.append(paths)

				continue

			rate = track.rate

			# Go through each instrument
			paths = dict()
			stem_audio = dict()
			for stem in ["bass", "drums", "other", "vocals"]:
				path = track_path + "_" + stem + ".wav"
				audio = track.targets[stem].audio
				soundfile.write(path, audio, rate, "PCM_16")
				stem_audio[stem] = audio
				paths[stem] = path

			# Add other instruments to form accompaniment
			acc_audio = np.clip(sum([stem_audio[key] for key in list(stem_audio.keys()) if key != "vocals"]), -1.0, 1.0)
			soundfile.write(acc_path, acc_audio, rate, "PCM_16")
			paths["accompaniment"] = acc_path

			# Create mixture
			mix_audio = track.audio
			soundfile.write(mix_path, mix_audio, rate, "PCM_16")
			paths["mix"] = mix_path

			diff_signal = np.abs(mix_audio - acc_audio - stem_audio["vocals"])
			print("Maximum absolute deviation from source additivity constraint: " + str(np.max(diff_signal)))# Check if acc+vocals=mix
			print("Mean absolute deviation from source additivity constraint:    " + str(np.mean(diff_signal)))

			samples.append(paths)

		subsets.append(samples)

	return subsets

def createSATBDataset(model_config):

	tracks = dict()
	resampling_fs = model_config['expected_sr']

	# Get all the stem files
	tracks['train'] = glob.glob(model_config['satb_path_train']+"/**/*.wav",recursive=True) 
	tracks['valid'] = glob.glob(model_config['satb_path_valid']+"/**/*.wav",recursive=True)                                                              # Get entirety of training set
	#songs = set([os.path.splitext(os.path.basename(i))[0].split('_')[1] for i in dsd_train])                                # Extract only songs from name
	# val_songs = random.sample(songs, int(np.ceil(0.2*len(songs))))                                                          # Take 20% of all the songs in training set
	# val_idx   = [int(i) for i, track in enumerate(dsd_train) if [True for song in val_songs if str('_'+song+'_') in track]] # get indices of all validation song stems
	# train_idx = [i for i in range(len(dsd_train)) if i not in val_idx]                                                      # Get all files from training set that are not part of valid set

	# tracks['train'] = np.take(dsd_train, train_idx)
	# tracks['valid'] = np.take(dsd_train, val_idx)
	#tracks['test']  = glob.glob(model_config["satb_path_test"]+"/*.wav",recursive=False)

	h5file = h5py.File(model_config["satb_hdf5_filepath"], "w")

	# Write stems to hdf5 file for train/valid/test partitions
	for curr_partition in ["train", "valid"]:

		print("Writing " + curr_partition + " partition with "+str(len(tracks[curr_partition]))+" files...")

		# Shuffle sample order
		stem_list = tracks[curr_partition]
		random.shuffle(stem_list)

		# Create group for set
		if not str(curr_partition) in h5file:
			set_grp  = h5file.create_group(curr_partition)

		count = 0
		print('Ready to create HDF5') 
		for track in stem_list:

			count += 1

			filename = os.path.splitext(os.path.basename(track))[0].split('_')
			print('processing '+str(count)+' of '+str(len(stem_list))+' files')

			song = filename[1]
			part = ''.join(filename[2:4])

			# Create group for the PART if needed
			if not str(curr_partition+'/'+part) in h5file:
				part_grp  = set_grp.create_group(part)

			# Create group for the SONG if needed
			if not str(curr_partition+'/'+part+'/'+song) in h5file:
				part_grp = h5file[str(curr_partition+'/'+part)]
				subgrp  = part_grp.create_group(song)

			# Once part groups / song subgroups are created, store file
			audio, s = librosa.load(track, sr=resampling_fs)
			subgrp = h5file[str(curr_partition+'/'+part+'/'+song)]
			subgrp.create_dataset("raw_wav",data=audio)

	print('Done Creating HDF5') 
	h5file.close()

def SATBBatchGenerator(hdf5_filepath, batch_size, num_frames, use_case=0, partition='train', resampling_fs=22050, debug=False):

    dataset   = h5py.File(hdf5_filepath, "r")
    sources = ['soprano','tenor','bass','alto']
    itCounter = 0
    allParts  = []
    partPerSongDict = {}

    # 1 Create dict of available parts per songs
    allParts = [key for key in dataset[partition].keys()]
    
    for part in allParts:
        songs = dataset[partition][part]
        for song in songs:
            if song in partPerSongDict:
                partPerSongDict[song].append(part)
            else:
                partPerSongDict[song] = [part]

    # List numbers of singer per part for a given song
    partCountPerSongDict = {}
    for song in partPerSongDict.keys():
        parts = partPerSongDict[song]
        parts = [x[:-1] for x in parts]
        partCountPerSongDict[song] = {i:parts.count(i) for i in parts}

        # If a part is missing from the song, add its key and 0 as part count
        diff = list(set(sources) - set(partCountPerSongDict[song].keys()))
        if len(diff) != 0:
            for missing_part in diff:
                partCountPerSongDict[song][missing_part] = 0

    while True:

        itCounter = itCounter + 1
        #print('Iteration '+str(itCounter))

        randsong = random.choice(list(partCountPerSongDict.keys()))

        # Get all available part from chosen song
        part_count = partCountPerSongDict[randsong]

        startspl = 0
        endspl   = 0

        out_shape  = np.zeros((batch_size, num_frames,1))
        out_shapes = {'soprano':np.copy(out_shape),'alto':np.copy(out_shape),'tenor':np.copy(out_shape),'bass':np.copy(out_shape), 'mix':np.copy(out_shape)}

        for i in range(batch_size):

            # Use-Case: At most one singer per part
            if (use_case==0):
                max_num_singer_per_part = 1
                randsources = random.sample(sources, random.randint(1,len(sources)))                   # Randomize source pick if at most one singer per part
            # Use-Case: Exactly one singer per part
            elif (use_case==1):
                max_num_singer_per_part = 1
                randsources = sources                                                                  # Take all sources + Set num singer = 1
            # Use-Case: At least one singer per part
            else:
                max_num_singer_per_part = 4
                randsources = sources                                                                  # Take all sources + Set max num of singer = 4 


            # Get Start and End samples. Pick random part to calculate start/end spl
            while startspl == 0:
                try:
                    randpart = random.choice(sources) + '1'
                    startspl = random.randint(0,len(dataset[partition][randpart][randsong]['raw_wav'])-num_frames) # This assume that all stems are the same length
                except:
                    continue


            endspl   = startspl+num_frames

            # Get Random Sources: 
            randsources_for_song = [] 
            for source in randsources:
                # If no singer in part, default it to one and fill array with zeros later
                if part_count[source] > 0:
                    max_for_part = part_count[source] if part_count[source] < max_num_singer_per_part else max_num_singer_per_part
                else:
                    max_for_part = 1 

                num_singer_per_part = random.randint(1,max_for_part)                      # Get random number of singer per part based on max_for_part
                singer_num = random.sample(range(1,max_for_part+1),num_singer_per_part)   # Get random part number for the number of singer based off max_for_part
                randsources_for_part = np.repeat(source,num_singer_per_part)              # Repeat the parts according to the number of singer per group
                randsources_for_part = ["{}{}".format(a_, b_) for a_, b_ in zip(randsources_for_part, singer_num)] # Concatenate strings for part name
                randsources_for_song+=randsources_for_part

            # Retrieve the chunks and store them in output shapes 
            zero_source_counter = 0                                        
            for source in randsources_for_song:

                # Try to retrieve chunk. If part doesn't exist, create array of zeros instead
                try:
                    source_chunk = dataset[partition][source][randsong]['raw_wav'][startspl:endspl]              # Retrieve part's chunk
                except:
                    zero_source_counter += 1
                    source_chunk = np.zeros(num_frames)

                out_shapes[source[:-1]][i] = np.add(out_shapes[source[:-1]][i],source_chunk[..., np.newaxis])# Store chunk in output shapes
                out_shapes['mix'][i] = np.add(out_shapes['mix'][i],source_chunk[..., np.newaxis])            # Add the chunk to the mix
            
            # Scale down all the group chunks based off number of sources per group
            scaler = len(randsources_for_song) - zero_source_counter
            out_shapes['soprano'][i] = (out_shapes['soprano'][i]/scaler)
            out_shapes['alto'][i]    = (out_shapes['alto'][i]/scaler)
            out_shapes['tenor'][i]   = (out_shapes['tenor'][i]/scaler)
            out_shapes['bass'][i]    = (out_shapes['bass'][i]/scaler)
            out_shapes['mix'][i] = (out_shapes['mix'][i]/scaler)
        
        # if debug == True, only proceed for 16 iterations
        if debug==True and itCounter<16:

            if partition=='train':
                rand_pick = random.randint(0,batch_size-1)
                debug_dir = './debug/it#'+str(itCounter)+'_batchpick#'+str(rand_pick)
                if not os.path.isdir(debug_dir):
                    os.mkdir(debug_dir)
                for source in sources:
                    soundfile.write(debug_dir+'/'+source+'.wav', out_shapes[source][rand_pick], resampling_fs, 'PCM_24')
                soundfile.write(debug_dir+'/'+'mix'+'.wav', out_shapes['mix'][rand_pick], resampling_fs, 'PCM_24')

        yield out_shapes


def getCCMixter(xml_path):
	tree = etree.parse(xml_path)
	root = tree.getroot()
	db_path = root.find("./databaseFolderPath").text
	tracks = root.findall(".//track")

	samples = list()

	for track in tracks:
		# Get mix and vocal instruments
		voice = get_path(db_path, track.xpath(".//instrument[instrumentName='Voice']")[0])
		mix = get_path(db_path, track.xpath(".//instrument[instrumentName='Mix']")[0])
		acc = get_path(db_path, track.xpath(".//instrument[instrumentName='Instrumental']")[0])

		samples.append({"mix" : mix, "accompaniment" : acc, "vocals" : voice})

	return samples

