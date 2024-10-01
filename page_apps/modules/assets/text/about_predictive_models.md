## Predictions

The predictions are very much just for fun and not to be taken seriously for guiding purchasing decisions. You know what coffees you've liked in the past. The best help this site can provide is to make it easier to find similar coffees going forward. 

Do not use these predictions as a purchasing guide. Disclaimers out of the way, here is how the predictions are generated.

### Training Data
Reviews of *roasted* coffees have been gathered from [coffeereview.com](https://www.coffeereview.com)

Coffee Reivew recommends interpreting their scores in, roughly, 7 buckets, as defined by their [interpretation guide](https://www.coffeereview.com/interpret-coffee/). 

We trained our rating predictor by looking at the country (or countries) of origin listed in the review and the 'Blind Assessment' where Coffee Review puts its detailed tasting notes. The Blind Assessment is encoded into a 768 dimension embedding as part of the training. The model is a logistic regression, trained over 7700+ reviews. 7 is the highest range on the Coffee Review scale, 1 is the lowest range.

For predictions we feed the same information, country of origin and encoded flavor descriptions, into the model.

We'll retrain and refine the model as we go, incorporating more detailed origin information, and cleaned up flavor descriptions. In particular, we're working on homogenizing the flavor descriptions in the training data to be a neater fit to the kind of tasting notes you find from green sellers.

### Caution

Given that the predictive model comes from Coffee Review and is in no way grounded in cupping scores, do not take this information as a predicted cupping score. We've not included the raw Coffee Review ratings here so as to further discourge such a mistake and are only including the rating 'bucket' where 7 is the highest and 1 is the lowest score range.

Similarly, a green coffee flavor description is not the same as a roasted coffee flavor description. The assumption underlying this model is that if you are the kind of highly skilled roaster that sends their coffees into Coffee Review, you could bring the flavors listed in the green coffee seller's flavor notes out in the best possible ways. 

Again, it's important to emphasize that the quality prediction ratings here are based on the Coffee Review reviews. The model is predicting what *Coffee Review* will think of a rather than some objective marker of coffee quality.

### Let the fun begin

As a kind of proof that the model is latching onto patterns that could be anticipated, have a look at the Aggregate Country Results and Aggregate Flavor Results tabs above. They give some good support for the model!

When we look at countries and flavors that are most strongly tied to different quality buckets, we see trends that are prima facie reasonable given trends in coffee taste and opinion. For example, roasty flavors show up most prominently in lower rated coffeed. Floral, sweet flavors show up in higher rated coffees. Similarly, when it comes to countries, Brazil and India tend to be most strongly linked to lower rated coffees. Kenya, Ethiopia and some Central American coffees are more strongly linked to higher rated coffees.

We measure the strength of connection between a flavor or a coffee through a TF-IDF. Intuitively, the idea is that a TF-IDF score for a country or flavor relative to a particular quality score (1-7) is higher when the country or flavor occurs more frequently in that quality bucket than it does in other quality buckets.

One last note: so few coffees score in the 7 range (both in the predictions made about the green coffees for sale and in the original coffee reviews) that it's not worth thinking too hard about which countries or flavor notes stand out as tied to the 7 range.