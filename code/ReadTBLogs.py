import argparse
import sys
import json
from os import listdir
from os.path import isfile, join
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
event_acc = EventAccumulator('/home/jakemdaly/Documents/courses/cse-291/pa2/data/LOG_RNN_PTB_50_TRIALS/logs/BS=32_LR=0.001_EB=250_RNN_HS=384_L=3_BI=0_LS=48_WD=0.05_ANN=SIGMOID_KLT_TS=2020-May-20-20_58_08')
event_acc.Reload()
# Show all tags in the log file
# print(event_acc.Tags())

# E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
w_times, step_nums, vals = zip(*event_acc.Scalars('TRAIN-Epoch/ELBO'))
w_times, step_nums, vals = zip(*event_acc.Tensors('args/text_summary'))

# print(vals[0].string_val[0].decode('utf-8')[9:])
# splits = vals[0].string_val[0].decode('utf-8').split(", ")
# for s in splits:



def main():

	data_dir = '/home/jakemdaly/Documents/courses/cse-291/pa2/data/'
	
	loggers = []
	for experiment in listdir(data_dir):
	# for experiment in ['LOG_GRU_TWITTER_50_TRIALS']:
		if 'VAN' in experiment: 
			print("Skipping VAN")
			continue
		CurrentLogger = Logger()
		path = data_dir + experiment + '/logs/'
		files = listdir(path)
		for i, file in enumerate(files):
			print("Finished... %s/%s"%(i, len(files)))
			f = path + file
			try:
				CurrentLogger.TrainMetrics.AddDataAndSummary(f)
				CurrentLogger.ValidMetrics.AddDataAndSummary(f)
			except:
				print("Exception occurred while adding data for metrics contained in file %s"%(path+file))
			# if i == : break
		with open(experiment+'.json', 'w') as out:
			JSON_DUMP = {
			'Train': CurrentLogger.TrainMetrics.data,
			'Valid': CurrentLogger.ValidMetrics.data
			}
			json.dump(JSON_DUMP, out)
			# json.dump(CurrentLogger.ValidMetrics.data, out)
	



class Logger:

	def __init__(self):
		self.TrainMetrics = Metrics("TRAIN")
		self.ValidMetrics = Metrics("VALID")



class Metrics:

	def __init__(self, train_or_valid):
		
		assert train_or_valid in ("TRAIN", "VALID"), "Invalid train_or_valid"
		self.train_or_valid = train_or_valid
		self.data = []
		# self.text_summary = []

		self.tags = [f'{self.train_or_valid}/ELBO',
	 		f'{self.train_or_valid}/NLL_Loss', 
	 		f'{self.train_or_valid}/KL_Loss', 
	 		f'{self.train_or_valid}/KL_Weight', 
	 		f'{self.train_or_valid}-Epoch/ELBO']

	def AddDataAndSummary(self, file_name):
		self.Add(file_name)
		# self.text_summary.append(self._GetSummary(file_name))
		

	def Add(self, file_name):
		
		ae, rl, kl, kw, ee = self._GetAllVals(file_name)
		vals = self._GetSummary(file_name)
		opts = vals[0].string_val[0].decode('utf-8')[9:]
		# f = file_name.split('/')[-1]
		temp_dict = {
		'opts': opts,
		'all_elbo': ae,
		'recon_loss': rl,
		'KL_loss': kl,
		'KL_weight': kw,
		'epoch_elbo': ee
		}

		self.data.append(temp_dict)

	def _GetAllVals(self, file):

		event_acc = EventAccumulator(file).Reload()
		print(file)
		_, _, ae = zip(*event_acc.Scalars(self.tags[0]))
		_, _, rl = zip(*event_acc.Scalars(self.tags[1]))
		_, _, kl = zip(*event_acc.Scalars(self.tags[2]))
		_, _, kw = zip(*event_acc.Scalars(self.tags[3]))
		_, _, ee = zip(*event_acc.Scalars(self.tags[4]))
		return ae[-1], rl[-1], kl[-1], kw[-1], ee[-1]

	def _GetSummary(self, file):

		event_acc = event_acc = EventAccumulator(file).Reload()
		_, _, vals = zip(*event_acc.Tensors('args/text_summary'))
		return vals

# class Summary:

# 	def __init__(self, summary):
# 		args



if __name__ == '__main__':
	main()