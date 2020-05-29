import time
import torch
print(torch.__version__)
import numpy as np
import json
import argparse
import logging
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
from collections import OrderedDict, defaultdict
from sklearn.manifold import TSNE

from ptb import PTB
from PoliticianTweetsDataset import PoliticianTweets
from utils import to_var, idx2word, experiment_name
from model import SentenceVAE

logging.basicConfig(
	format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
	datefmt="%m/%d/%Y %H:%M:%S",
	level=logging.INFO,
)
logger = logging.getLogger(__name__)

def main(args):

	ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

	# The Twitter user who we want to get tweets from
	BS = "SenSanders"
	EW = "ewarren"
	AOC = "AOC"
	HC = "HillaryClinton"
	MM = "senatemajldr"
	LG = "LindseyGrahamSC"
	DT = "realDonaldTrump"
	DD = "SenatorDurbin"
	JM = "Sen_JoeManchin"
	JT = "SenatorTester"
	MR = "MittRomney"
	KM = "GOPLeader"
	DC = "RepDougCollins"
	CS = "SenSchumer"
	CB = "cbellantoni"
	EE = "ewerickson"
	MF = "mindyfinn"
	GG = "ggreenwald"
	NP = "nicopitney"
	TC = "TPCarney"
	AC = "anamariecox"
	DB = "donnabrazile"
	TCar = "TuckerCarlson"
	politicians = [BS, EW, AOC, HC, MM, LG, DT, DD, JM, JT, MR, KM, DC, CS, CB, EE, MF, GG, NP, TC, AC, DB, TCar]
	partial_splits = ["test."+pol for pol in politicians]

	# Test splits are generated from GetTweets.py
	splits = ["train", "valid"] + partial_splits

	datasets = OrderedDict()
	for split in splits:

		if args.dataset == 'ptb':
			Dataset = PTB
		elif args.dataset == 'twitter':
			Dataset = PoliticianTweets
		else:
			print("Invalid dataset. Exiting")
			exit()

		datasets[split] = Dataset(
			data_dir=args.data_dir,
			split=split,
			create_data=args.create_data,
			max_sequence_length=args.max_sequence_length,
			min_occ=args.min_occ
		)

	# Must specify the pickle file from which to load the model
	if args.from_file != "":
		model = torch.load(args.from_file)
		checkpoint = torch.load("/home/jakemdaly/PycharmProjects/vae/pa2/Language-Modelling-CSE291-AS2/bin/2020-May-26-06:03:46/E2.pytorch")
		model.load_state_dict(checkpoint)
		print("Model loaded from file.")
	else:
		print("Must be initialized with a pretrained model/pickle file. Exiting...")
		exit()

	if torch.cuda.is_available():
		model = model.cuda()

	print(model)

	# These are the dictionaries that get dumped to json
	PoliticianSentences = {}
	PoliticianLatents = {}
	for split in splits[2:]:
		PoliticianLatents[split] = None
		PoliticianSentences[split] = None

	for epoch in range(args.epochs):

		for split in splits[2:]:

			data_loader = DataLoader(
				dataset=datasets[split],
				batch_size=args.batch_size,
				shuffle=split == 'train',
				num_workers=0,
				pin_memory=torch.cuda.is_available()
			)

			# Enable/Disable Dropout
			if split == 'train' or split == 'valid':
				continue
			else:
				model.eval()

			for iteration, batch in enumerate(data_loader):

				# Get kth batch
				for k, v in batch.items():
					if torch.is_tensor(v):
						batch[k] = to_var(v)

				# Latent variables
				try:
					if PoliticianLatents[split] is None: PoliticianLatents[split] = get_latent(model, batch['input'], batch['length']).data.numpy()
					else:
						# pesky situation: if it has only one dimension, we need to unsqueeze it so it can be appended.
						if len(np.shape(get_latent(model, batch['input'], batch['length']).data.numpy())) == 1 and np.shape(get_latent(model, batch['input'], batch['length']).data.numpy())[0] == args.latent_size:
							PoliticianLatents[split] = np.append(PoliticianLatents[split], np.expand_dims(get_latent(model, batch['input'], batch['length']).data.numpy(), 0), axis=0)
						else:
							PoliticianLatents[split] = np.append(PoliticianLatents[split], get_latent(model, batch['input'], batch['length']).data.numpy(), axis=0)
				except:
					print(split)
				# # Sentences corresponding to the latent mappings above
				if PoliticianSentences[split] is None:
					PoliticianSentences[split] = idx2word(batch['input'].data, i2w=datasets['train'].get_i2w(), pad_idx=datasets['train'].pad_idx)
				else:
					PoliticianSentences[split].append(idx2word(batch['input'].data, i2w=datasets['train'].get_i2w(), pad_idx=datasets['train'].pad_idx))

	# Dump all data to json files for analysis
	for split in splits[2:]:
		PoliticianLatents[split] = PoliticianLatents[split].tolist()
	with open("PoliticianLatentsE2.json", 'w') as file:
		json.dump(PoliticianLatents, file)
	with open("PoliticianSentences.json", 'w') as file:
		json.dump(PoliticianSentences, file)
	print("Done.")


def get_latent(model, input_sequence, length):
	batch_size = input_sequence.size(0)
	sorted_lengths, sorted_idx = torch.sort(length, descending=True)
	input_sequence = input_sequence[sorted_idx]

	# ENCODER
	input_embedding = model.embedding(input_sequence)

	packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

	_, hidden = model.encoder_rnn(packed_input)

	if model.bidirectional or model.num_layers > 1:
		# flatten hidden state
		hidden = hidden.view(batch_size, model.hidden_size * model.hidden_factor)
	else:
		hidden = hidden.squeeze()

	# REPARAMETERIZATION
	mean = model.hidden2mean(hidden)
	logv = model.hidden2logv(hidden)
	std = torch.exp(0.5 * logv)

	return mean


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--data_dir', type=str, default='data')
	parser.add_argument('--create_data', action='store_true')
	parser.add_argument('--max_sequence_length', type=int, default=60)
	parser.add_argument('--min_occ', type=int, default=1)
	parser.add_argument('--test', action='store_true')
	parser.add_argument('-ds', '--dataset', type=str, default='ptb')
	parser.add_argument('--from_file', type=str, default="")

	parser.add_argument('-ep', '--epochs', type=int, default=10)
	parser.add_argument('-bs', '--batch_size', type=int, default=32)
	parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

	parser.add_argument('-eb', '--embedding_size', type=int, default=300)
	parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
	parser.add_argument('-hs', '--hidden_size', type=int, default=256)
	parser.add_argument('-nl', '--num_layers', type=int, default=1)
	parser.add_argument('-bi', '--bidirectional', action='store_true')
	parser.add_argument('-ls', '--latent_size', type=int, default=16)
	parser.add_argument('-wd', '--word_dropout', type=float, default=0)
	parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)

	parser.add_argument('-af', '--anneal_function', type=str, default='identity')
	parser.add_argument('-ag', '--anneal_aggression', type=float, default=1.0)
	parser.add_argument('-klt', '--kl_threshold', type=float, default = -1.0)

	parser.add_argument('-v','--print_every', type=int, default=50)
	parser.add_argument('-tb','--tensorboard_logging', action='store_true')
	parser.add_argument('-log','--logdir', type=str, default='logs')
	parser.add_argument('-bin','--save_model_path', type=str, default='bin')

	args = parser.parse_args()

	args.rnn_type = args.rnn_type.lower()
	args.anneal_function = args.anneal_function.lower()

	assert args.rnn_type in ['rnn', 'lstm', 'gru']
	assert 0 <= args.word_dropout <= 1

	main(args)
