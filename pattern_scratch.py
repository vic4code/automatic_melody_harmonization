import numpy as np

#### SIA algorithm
class SIA():
    """ 
    Find repeated patterns.

    Parameters
    ----------
    input_seq : List [time,pitch].

    Returns
    -------
    table : sorted list with all vectors.
    """
    def __init__(self,input_seq):
        self.input_seq = sorted(input_seq)
        self.start = self.end = self.input_seq
        self.table = []
        self.vectors = []
        self.count_dict = {}
        self.length = len(self.input_seq)
        
        ## Find vector table
        # row
        i = 1
        # col
        j = 0
        
        # Create table
        while j < self.length:
            i = j + 1
            while i < self.length:
                vector = [self.end[i][0]-self.start[j][0], self.end[i][1]-self.start[j][1]]
    #             print(vector,i,j,start[j])
                self.vectors.append(vector)
                self.table.append([vector,self.start[j]])
                i += 1
            j += 1

        self.table = sorted(self.table)
        self.count_dict = {str(i):self.vectors.count(i) for i in self.vectors}
    
    ## Calculate max pattern
    def max_pattern(self):
        if bool(self.count_dict):
            max_value = max(self.count_dict.values())
            ratio = max_value / (self.length - 1)
        else:
            max_value = 0
            ratio = 1
        
        return max_value, ratio

### Find beats in a sequence
def find_beats(arr):
    length = len(arr)
    current = 0
    beats = []
    count = 0
    i = 0 

    while i < length:
        if arr[i] == arr[current]:
            count += 1
            i += 1
        else:
            beats.append(count)
            current = i
            count = 0
    
    # Append the last
    beats.append(count)
    return beats    

#### Pitch ratio
# pitch_patterns = SIA(input_seq)
# _, R_pitch = pitch_patterns.max_pattern()


#### Rhythm
# rhythm_patterns = SIA(input_seq)
# _, R_rhythm = rhythm_patterns.max_pattern()

#### Tonal language 
def tonal_explainable():
    """ 
    Check if chord subsequence is explainable by the following scale:

    Major scale
    Minor scale
    Mode scale

    Parameters
    ----------

    Returns
    -------
    boolean : True or False.
    """
    #### R_Tonal
    input_seq = []

    tonal_regions = []
    seq_length = len(input_seq)
    head = 0
    rear = 1

    # Search if chord sequence is tonally explanable or not
    while rear < seq_length:
        sub_seq = input_seq[head:rear]

        if not tonal_language(sub_seq):
            tonal_regions.append([head,rear])
            head = rear
            rear += 1

        else:
            rear += 1

    # Merge overlaped regions
    merged_tonal_regions = []
    temp_merge = [0,0]
    flag = 0

    for i in range(len(tonal_regions) - 1):

        if tonal_regions[i][1] == tonal_regions[i+1][0]:
            temp_merge[0] = tonal_regions[i][0]
            temp_merge[1] = tonal_regions[i+1][1]
            flag = 1

        else:
            if flag == 1:
                merged_tonal_regions.append(temp_merge)
                temp_merge = [0,0] 
            else:
                merged_tonal_regions.append(tonal_regions[i])
                flag = 0

    R_tonal = len(merged_tonal_regions)/len(input_seq)   
    
    return True

