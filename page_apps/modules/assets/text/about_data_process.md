

### Step 1: Collecting the Raw text
Data is collected directly from green seller websites. The initial phase of data collection is the simplest. After manually reviewing the structure of a green seller's website, we determine the smallest chunk of a given coffee's webpage that contains the information about the coffee. So, we're ignoring the 'You might also like:' recommendations, the links to other parts of the seller's website, and then menu portions of the website. We take just the raw text from this part of the page, ditching all of the html and javascript structure.

### Step 2: Extracting Structured Information from Raw Text
The second phase of the data collection process involves using an LLM to parse the text from the website into a structured format that we can use. The LLM is asked to find:


- A name for the coffee
- The country of origin for the coffee
- The region of the country that the coffee came from
- Any micro location information about where the coffee came from that is smaller than the region, such as the mill, town, farm, etc.
- Any information about processing method and fermentation
- Any information about the varietals of coffee 
- A list of flavor notes
- A score, if present

In each case, if the LLM can't determine an appropriate answer to these, it responds with 'UNKNOWN'.

For consistency in output and avoiding the LLM making up content, we've used Anthropic's Claude 3 models with tool use here.

In addition to the information extracted from the text of the page, we also track the following information for each coffee:


- The date we first observed it on the seller's website
- Whether or not the coffee continues to exist on the seller's website, or has sold out
- The URL for the particular page that contains the coffee information
- A name for the coffee based on the URL for its particular page
- Which coffee seller sells this coffee.

### Step 3: Data Preprocessing Step 1, creating a minimum usable dataset
There are two main causes of useless information while trying to find data about green coffee in this automated way. 

The first cause of useless information is the lack of information on green seller's own webiste. This can happen in many ways. First, many green sellers sell things other than coffees, even on their pages dedicated to selling coffee! We do our best to avoid even scraping these pages but some simply can't be avoided in a programmatic way. Second, especially with decaf coffees and preblended green coffees, there is essentially no interesting or useful information abou the coffee other than its country or countries of origin and the decaf processing method. Third, sometimes there simply isn't much information for a regular coffee on a site. Just a price and a country of origin.

The second cause of useless information happens when the website information can't be parsed in a useful way by the LLM. This can be because the structure of the website is unfriendly to looking at the raw text, that there's simply too much information for the LLM to comfortably work through, or some other reason.

So, unfortunately, some coffees will have to be pruned at this point. Here is the process of pruning out the unusuable results of our initial scraping process:

1. Remove any coffees for whic the LLM did not return a response formatted as a JSON, or any response at all.
2. In over 98% of cases, the LLM returns a correctly formatted JSON. Sometimes, however, it slightly misnames a json field name (e.g. 'Sub-Region' instead of 'Subregion') (future revisions to the tool use requirement in the prompting will eliminte this issue entirely). So the next step is merging data fields that contain the same information but are slightly misnamed. 
3. Since we're going to be doing some quality predictions based on the information we're going to set a minimum standards requirement here. We're going to require that every coffee in our database has at least 3 of the following pieces of infomration: Name, Country, Processing method, Altitude, Flavor notes. So, for example, any coffee with a Name, Country of origin and Flavor notes will move on.
4. At a later time we might move this step to the deeper data processing methdos described above, but as far as the Varietal information goes, we simply do some simple cleaning of the text to remove any artifcats either from the original website or introduced through the LLM processing of the website text. The result is a clean list of varietals as described directly on the seller's webiste.

### Step 4: Homogenizing Data
The data extracted from seller websites in the processes described above arrives at this point in a wildly variable state.

Each green seller has their own approach to describing coffees, with varying level of details, varying depth of description on flavor. Different sellers have different ideas about what constitutes a coffee growing region within a country.  And, of course, there will be subtle and not so subtle spelling variations and errors made on seller websites.

On top of that, the LLM processing of the websites is itself a source of variety in the data. Sometimes the LLM might categorize a bit of location information as information about the subregion within a country. Other times it might categorize that same information as belonging in the micro-location category. 

In order to make the data usable, even for sorting purposes, it's essential to try to impose some standardization. At this stage we use the LLM again, this time with very strict checks and feedback loops to produce the following homogenization:

#### Location Homogenization
All location information—county, region, micro—along with the name of the coffee, is passed to the LLM. The LLM is tasked with choosing a country (from a fixed list of countries), region (from a fixed list of regions for that country), and remainder location information for each coffee. We need some way to standardize regions for this purpose. The choice here is somewhat arbitrary and obviously taste and understanding on these topics change over time. For a starting place, we've used *The World Atlas of Coffee*, second edition (insert link).

#### Altitude Homogenization
Some altitude information is a single number, some is a range,some is in feet, some is in meters. We've converted all altitude information to a range of meters above sea level. Each coffee has a lower bound elevation, an upper bound elevation, and a units of measure (masl)

#### Process Homogenization
Process descriptions are wildly in flux and come in an incredible variety. To impose some fixed structure here, we break down the LLM task into two parts. First, find the process type (from a fixed set of choices). Second, see if there was some fermentation involved. Of course, what counts as fermentation in such a way as to be called a coffee that has been processed with fermentation is wildly controversial. 

The set of process choices that we restrcited our LLM to choose from are: Washed, Natural, Honey, Wet Hulled, Monsoon, Decaf (obviously decaf isn't like the others in terms of process; this is mostly just a way to have a convenient tool for identifying decafs that weren't otherwise caught in the data preprocessing stages)

Whether a coffee is classified as fermented or not to our LLM depends on whether the texts describing its process uses phrases like anaerobic, carbonic, maceration, yeast, co-fermented, multi-stage.

Because we know that people are very sensitive to and about process details you also have the option to view the original text of the process description on an individual coffee page.

#### Flavor Homogenization
As an experimental feature, and to serve as a basis for our flavor representation graphics on each coffee's individual listing page, we have also taken some effort to do flavor note homogenization.

As with regions, how exactly to standardize flavor descriptions is quite a task. If it plays more of a role in our quality prediction process, we'll make a note of it. For now, all quality prediction is handled by the original flavor descriptions. Here is how we homogenize flavor descriptions.

- First, all flavor descriptions are broken into a list by our initial LLM processing of the green seller's web page for the coffee.
- Second, each flavor is mapped onto the SCAA flavor wheel. 
    - Here is the version of the wheel we start with: link_to_image. We will make some modifications for consistency as described below. 
    - To map a flavor to a wheel we assign each flavor description to a flavor family (as defined by the inner most ring on the flavor wheel), a flavor genus (as defined by the middle ring on the flavor wheel), and a flavor species (as defined by the outer ring on the flavor wheel).
    - To add consistency and also allow the LLM to function more reliably we've made two minor tweaks to the flavor wheel. First, we ensure that all flavor outcomes have the same depth. Sometiems this means simply repeating a genus as a species if the genus has no species. Second, we've added some generic categories to the list of species as catch alls. For example, since so many coffees are described as having a berry like flavor but not any type of berry in particular, we've added a 'generic berry' flavor species.

Since this process is more artificia than the others in imposing a structure on unstructured flavor description text, and also because any flavor notes that describe texture more than flavor will be treated as being about flavor (ie syrupy flavor ntoes will be categoriezed as molasses or carmelized flavors even though they may instead be about the texture), we will always display the . For fun, you can also always seen how each individual flavor note has been taxonomized. You also see the result of this taxonomization in the flavor graphic depictions on each individual coffee's page. We start with a base wheel and darken each family, genus and species for each flavor note in the flavor list.