# #BringbackFranku  

  
![](https://i.ytimg.com/vi/lHzMKHrqXSc/maxresdefault.jpg)  
  
This project aims to produce an AI that can generate new Filthy Frank scripts with a stretch goal of having those scripts synthesized with Franku's voice.  

   
# Goals  

- Evaluate the best NLP framework for the job 
- Test this initially on movie/music scripts. 
- Collate all the Filthy Frank episode dialogue into text files.  
- Generate new scripts.  
- Work on voice synthesis 

## Low Level   
  
Once the initial scripts have been colalted from using `youtube` -> `captions` -> `edit` and saving to file: sentences will need to be tokenised. Most NLP processes work best on smaller chunks rather than episodes at a time. The challenge will setting delimiters, token values, and associated meta-data.  

## Potential MetaData design   

| Episode        | Length           | special words  | characters | languages   | text  |
| ------------- |:-------------:| -----:|:-------------:|:-------------:|:-------------:|
| PINK GUY COOKS FRIED RICE AND RAPS   | 14      | *slurp noises*     |   pink guy  |  English, Japanese  |    I ain't no Gordon Ramsey but I make the best damn rice ya pansies. | 

 

# Tools 
  
A list of suggested tools to work with.  
  
- word2vec
- nltk
- Spacy for preprocessing and tokenisaiton
- LSTM keras 


Links that have beeen helpfull  

https://www.kaggle.com/jrobischon/wikipedia-movie-plots/kernels    
https://juanitorduz.github.io/movie_plot_text_gen/   
