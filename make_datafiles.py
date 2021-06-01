import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2


dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

all_train_urls = "url_lists/all_train.txt"
all_val_urls = "url_lists/all_val.txt"
all_test_urls = "url_lists/all_test.txt"

cnn_tokenized_stories_dir = "cnn_stories_tokenized"
dm_tokenized_stories_dir = "dm_stories_tokenized"
finished_files_dir = "dataset"
chunks_dir = os.path.join(finished_files_dir, "chunked")

# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506

VOCAB_SIZE = 50000
CHUNK_SIZE = 5000 # num examples per chunk, for the chunked data


def chunk_file(set_name):
  in_file = 'dataset/%s.bin' % set_name
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join('dataset', '%s_%d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1


def chunk_all():
  # Make a dir to hold the chunks
  if not os.path.isdir(chunks_dir):
    os.mkdir(chunks_dir)
  # Chunk the data
  for set_name in ['train', 'val', 'test']:
    print("Splitting %s data into chunks..." % set_name)
    chunk_file(set_name)
  print("Saved chunked data in %s" % chunks_dir)


def tokenize_stories(stories_dir, tokenized_stories_dir):
  """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
  print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
  stories = os.listdir(stories_dir)
  # make IO list file
  print("Making list of files to tokenize...")
  with open("mapping.txt", "w") as f:
    for s in stories:
      f.write("%s \t %s\n" % (os.path.join(stories_dir, s), os.path.join(tokenized_stories_dir, s)))
  command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
  print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
  subprocess.call(command)
  print("Stanford CoreNLP Tokenizer has finished.")
  os.remove("mapping.txt")

  # Check that the tokenized stories directory contains the same number of files as the original directory
  num_orig = len(os.listdir(stories_dir))
  num_tokenized = len(os.listdir(tokenized_stories_dir))
  if num_orig != num_tokenized:
    raise Exception("The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
  print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))


def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines


def hashhex(s):
  """Returns a heximal formated SHA1 hash of the input string."""
  h = hashlib.sha1()
  h.update(s.encode())
  return h.hexdigest()


def get_url_hashes(url_list):
  return [hashhex(url) for url in url_list]


def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."


def get_art_abs(story_file):
  lines = read_text_file(story_file)

  # Lowercase everything
  lines = [line.lower() for line in lines]

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  lines = [fix_missing_period(line) for line in lines]

  # Separate out article and abstract sentences
  article_lines = []
  highlights = []
  next_is_highlight = False
  for idx,line in enumerate(lines):
    if line == "":
      continue # empty line
    elif line.startswith("@highlight"):
      next_is_highlight = True
    elif next_is_highlight:
      highlights.append(line)
    else:
      article_lines.append(line)

  # Make article into a single string
  article = ' '.join(article_lines)

  # Make abstract into a signle string, putting <s> and </s> tags around the sentences
  abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])

  return article, abstract


def write_to_bin(data_file, data,makevocab=False):

  if makevocab:
    vocab_counter = collections.Counter()
  with open(data_file+'/'+data+'.abstract','r') as f:
    source = f.readlines()
  with open(data_file+'/'+data+'.target.overlap','r') as f:
    present = f.readlines()
  with open(data_file+'/'+data+'.title','r') as f:
    absent = f.readlines()
  present = [SENTENCE_START+' ' + i.strip() + ' ' + SENTENCE_END for i in present]
  absent = [SENTENCE_START+' ' + i.strip() + ' ' + SENTENCE_END for i in absent]
  with open(data_file+'/test_summary_'+data+'.bin', 'wb') as writer:
    for idx in range(len(source)):
      
      article = source[idx]
      abstract = present[idx]
      abstract_abc = absent[idx]
      # Write to tf.Example
      tf_example = example_pb2.Example()
      tf_example.features.feature['article'].bytes_list.value.extend([article.encode()])
      tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode()])
      tf_example.features.feature['abstract_abc'].bytes_list.value.extend([abstract_abc.encode()])
      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))

      # Write the vocab to file, if applicable
      if makevocab:
        art_tokens = article.split(' ')
        abs_tokens = abstract.split(' ')
        abs_tokens_abc = abstract_abc.split(' ')
        abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
        abs_tokens_abc = [t for t in abs_tokens_abc if t not in [SENTENCE_START, SENTENCE_END]]
        tokens = art_tokens + abs_tokens + abs_tokens_abc
        tokens = [t.strip() for t in tokens] # strip
        tokens = [t for t in tokens if t!=""] # remove empty
        vocab_counter.update(tokens)

  print("Finished writing file %s\n" % data_file)

  # write vocab to file
  if makevocab:
    print("Writing vocab file...")
    with open(os.path.join(data_file, "vocab"), 'w') as writer:
      for word, count in vocab_counter.most_common(VOCAB_SIZE):
        writer.write(word + ' ' + str(count) + '\n')
    print("Finished writing vocab file")


def check_num_stories(stories_dir, num_expected):
  num_stories = len(os.listdir(stories_dir))
  if num_stories != num_expected:
    raise Exception("stories directory %s contains %i files but should contain %i" % (stories_dir, num_stories, num_expected))

data = ['inspec','krapivin','nus','semeval','kp20k']
#data = ['kp20k']
for i in data:
    write_to_bin('dataset', i, makevocab=False)
    chunk_file('test_summary_'+i)

#chunk_file('test_kp20k')
'''
if __name__ == '__main__':
  if len(sys.argv) != 3:
    print("USAGE: python make_datafiles.py <cnn_stories_dir> <dailymail_stories_dir>")
    sys.exit()
  cnn_stories_dir = sys.argv[1]
  dm_stories_dir = sys.argv[2]

  # Check the stories directories contain the correct number of .story files
  check_num_stories(cnn_stories_dir, num_expected_cnn_stories)
  check_num_stories(dm_stories_dir, num_expected_dm_stories)

  # Create some new directories
  if not os.path.exists(cnn_tokenized_stories_dir): os.makedirs(cnn_tokenized_stories_dir)
  if not os.path.exists(dm_tokenized_stories_dir): os.makedirs(dm_tokenized_stories_dir)
  if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

  # Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories
  tokenize_stories(cnn_stories_dir, cnn_tokenized_stories_dir)
  tokenize_stories(dm_stories_dir, dm_tokenized_stories_dir)

  # Read the tokenized stories, do a little postprocessing then write to bin files
  write_to_bin(all_test_urls, os.path.join(finished_files_dir, "test.bin"))
  write_to_bin(all_val_urls, os.path.join(finished_files_dir, "val.bin"))
  write_to_bin(all_train_urls, os.path.join(finished_files_dir, "train.bin"), makevocab=True)

  # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
  chunk_all()
'''
