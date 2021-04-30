import argparse
import numpy as np
from tqdm import tqdm

## Markov chain
class markov_chain(): 
    def __init__(self,chord_seqs,states = [x for x in range(96)]):
        
        self.chord_seqs = chord_seqs
        self.states = states
        self.num_state = len(states) #number of states
        self.M = [[0]*self.num_state for _ in range(self.num_state)]
    
    # Input one seq
    def transition_probability(self,seq):
#         M = [[0]*self.num_state for _ in range(self.num_state)]
        
        # Convert seq to index seq
        index_seq = [self.states.index(i) for i in seq]
       
        for (i,j) in zip(index_seq,index_seq[1:]):
            self.M[i][j] += 1

        #now convert to probabilities:
        for row in self.M:
            s = sum(row)
            if s > 0:
                row[:] = [f/s for f in row]
    
    # Input one seq
    def create_transition_matrix_by_one_seq(self,seq):
        self.transition_probability(seq)
        return np.array(self.M)
    
    # Input seqs
    def create_transition_matrix_by_many_seqs(self):
        for seq in self.chord_seqs:
            self.transition_probability(seq)
        return np.array(self.M)
    
    # Input one seq
    def calculate_surprisingness(self,seq,t,TM):
        
        current = seq[t]
        i_ = self.states.index(current)

        previous = seq[t - 1]
        j_ = self.states.index(previous)

        if TM[i_][j_] == 0:
            surprisingness = -np.log(TM[i_][j_] + 1e-4)
        else:
            surprisingness = -np.log(TM[i_][j_])
            
        return surprisingness
    
    def create_surprisingness_seqs(self,all_data=False):
    
        surprisingness_seqs = []
        n = len(self.chord_seqs)
        states = [x for x in range(96)]
        
        if all_data:
            TM = self.create_transition_matrix_by_many_seqs().transpose()
            
        for i in tqdm(range(n)):
            seq = self.chord_seqs[i]
            N = len(seq)
            T = range(1,N)
            surprisingness_seq = [0]

            if all_data:
                for t in T:
                    surprisingness = self.calculate_surprisingness(seq,t,TM)
                    surprisingness_seq.append(surprisingness)
                
            else:
                for t in T:
                    TM = self.create_transition_matrix_by_one_seq(seq[:t]).transpose()
                    self.M = [[0]*self.num_state for _ in range(self.num_state)]
                    surprisingness = self.calculate_surprisingness(seq,t,TM)
                    surprisingness_seq.append(surprisingness)
                   
            surprisingness_seqs.append(surprisingness_seq)

        surprisingness_seqs = np.array(surprisingness_seqs)
        surprisingness_seqs = np.expand_dims(surprisingness_seqs,axis=-1)

        return surprisingness_seqs, TM
# #     

## Main
def main():
    ''' 
    Usage:
    python create_surprisingness.py  ///
    '''

    parser = argparse.ArgumentParser(description='Set configs to training process.') 
    
    parser.add_argument('-all_data', default=False) 
    parser.add_argument('-filename', type=str, required=True) 
    args = parser.parse_args()
    
    chord_seqs = np.load('./data/number_96.npy')
    surprisingness_seqs, TM = markov_chain(chord_seqs).create_surprisingness_seqs(args.all_data)
    np.save('./data/' + args.filename , surprisingness_seqs)
    
if __name__ == '__main__':
    main()

    
