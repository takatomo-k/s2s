        
class SpeakerTransform(object):
    def __init__(self, side, data, min_frequency=0):
        self.side=side
        self.spkr={}
        self.max_length=2
        for sent in data:
            spkr = sent.strip().split("@")[1]
            if spkr not in self.spkr:
                self.spkr.update({spkr:len(self.spkr)})
        #print(self.spkr)
    def __len__(self):
        return len(self.spkr)
    
    def __call__(self, inputs):
        return torch.ones((1)).type(torch.LongTensor)*self.spkr[inputs[0]]
