import GetOldTweets3 as got
from afinn import Afinn

afinn = Afinn()

tweetCriteria = got.manager.TweetCriteria().setUsername("realDonaldTrump").setMaxTweets(20)
tweets = got.manager.TweetManager.getTweets(tweetCriteria)

for t in tweets:
    print(t.date)
    print(t.text)
    print(afinn.score(t.text))
    print()

    
