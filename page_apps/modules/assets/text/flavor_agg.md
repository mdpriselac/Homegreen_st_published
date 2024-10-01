### Flavors and Ratings
In the table below, you can see how strongly connected flavors are to a given quality range. Using the TF-IDF metric described in the Predictve Model tab above, we've calculated how strongly each flavor is connected each bucket quality. For these purposes we're looking at the taxonomized flavors so we can get a homogenized data set to compare across coffee sellers. 

We've measured the TF-IDF of flavor families, flavor genuses, and flavor species independently from one another. That is, we've looked at which flavor families are distinctive to a given quality rating among only other flavor families, and the same for flavor genera and species. From there we've calculated the 15 most distinct flavors, across family, genus, and species for each quality bucket and rank ordered them from 1-15, most distinctive to least distinctive. 

Each quality bucket has its own flavor wheel filled in based on these 15 most distinctive flavors. The most distinctive flavors are filled in the darkest with no transparency. The least (among the 15 most distinctive) distinctive flavor for a quality bucket is the most transparent among the flavors that are filled in on the quality bucket's wheel.

As you can see in the tabs below, many of the trends you might expect in terms of quality and flavor are found in our predictions. For example, roasty flavors are more prevalent in the 1 ranked coffees. As you move up the table Fruity berry and citrus flavors become more prominent. As you move from 4 to 5 coffees, fruity species start to enter the picture more, rather than just genuses. Finally as you ascend through levels 5, 6 and 7 floral flavors and sweetness start to move to become more strongly associated with the ranking category.

Again, as with the country data, we can glean some small insights into the way the predictive model is working. It's important to remember here that the model is following patterns in the Coffee Review data set, not some objective standard of coffee quality. 

Finally, remember that there are so very few coffees scoring in the 7 range that we don't need to focus too closely on that bucket to draw too many conclusions.