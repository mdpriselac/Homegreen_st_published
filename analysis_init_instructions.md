# Overview

We're adding a new feature to our streamlit app about the green coffee market. In this feature we're going to be providing analytics and visualizations for information about how coffee flavors relate to coffee origins (countries and their subregions) and flavors. We're also going to provide information on how different coffee sellers describe coffe flavors.

All of the pieces of this project should go in their own subdirectory, `anlaytics`. The database access system should be in `analytics/db_access`, the analytics systems should be in the `analytics/processing` folder. The front end system should should be in the `analytics/frontend` folder. Store the data in appropriately named subdirectories to `analytics/data/`

# Background

We're working with a database that has all the information we need, we just need to extract the right information, perform some analytics, and then create the streamlit page with visualizations and information. You can find information about each of these steps in three different markdown files in this folder.

## Step 1: The database access system

For the first step, you need to create the tools necessary to access our database. Read the `accessing_data_for_analysis.md` file to understand how to acess the data we need for our system.

The database is a supabase database and we'll be accessing it with the python client. This places an extremely important restriction on our system: supabase does NOT allow you to execute sql code remotely. Instead, you have to use the query builder as it's built into the supabase python package. Look up the docs for that or search the internet if you need to know how to use it.

The `accessing_data_for_analysis.md` also has instructions on how our analysis systems will expect to receive the data. Make sure you understand that side of the pipeline clearly as well.

When you finish creating the database anaytics system, write an extremely clear detailed plan of the data that comes out of this system and how it can be accessed. Save it as `anaytics_ready_data_instructions.md`

## Step 2: The Analytics system

Next we need to create the analytics system. Read the `analysis_data_plan.md` plan for the key analytics we want to implement. As you read this double check that the format you're receiveing from the database access system is correct.

When you've completed the analytics system, write up a clear instruction set for how the data has been created and how it can be accessed. Save it as `frontend_ready_data_instructions.md`

## Step 3: The front end

The analytics we create will be displayed as one page in a multipage streamlit app. We've created a design plan for that page that you can read at `analysis_front_end_plan.md`. Check that out before starting to wrok on the front end.

Make sure that you clearly understand the nature of the data that will be served to the front end by the analytics system and how to access it by checking with the `analysis_data_plan.md` and, once you create the analytics system, the `frontend_ready_data_instructions.md` file.

# General Instructions

Overall we want to keep this system relatively simple and lightweight. This is mostly a home based project though we will want to be able to maintain it easily so each system should be nicely modularized and documented well. You may deviate from the plans laid out in files above when you believe you have a cleaner, more modern and simple system than has been planned for. Make sure to ask before deviating, however, and document any deviations. Make sure that any changes earlier in the system are accounted for later in the system.

# Running scripts

To test files, we're using UV to run and manage python packages in environments. So, to test, use the uv run functionality.

