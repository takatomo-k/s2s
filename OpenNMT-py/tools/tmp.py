from gtts import gTTS

idx=0
for i in ["シャッターが停電のため動かない","夫が病気になったという理由で解雇された","そこのスマホに夢中のあなた"]:
    res=gTTS(i,lang='ja')
    res.save(str(idx)+".mp3")
    idx+=1