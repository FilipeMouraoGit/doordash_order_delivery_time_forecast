<a id="readme-top"></a>
<div align="center">  <h3 align="center">Doordash Forecast Delivery Challenge</h3></div>

<!-- ABOUT THE PROJECT -->
# Project description
This is project is divided in 2 steps 
- Build a Machine Learning model for a forecast task;
- Create an application to make predictions using the training model;

## Forecast problem overview

In the moment that an order is placed on DoorDash, there is an expected time of delivery which is crucial for a good user experience.  

The training data is a subset of DoorDash deliveries made in 2015, each row corresponds to a one unique delivery and has the following columns:

- market_id -> City/Region in which Doordash operates;
- created_at -> Timestamp when the order was submitted;
- actual_delivery_time -> Timestamp when the order arrived to the consumer;
- store_id -> An id representing the restaurant which the order was made;
- store_primary_category -> Cuisine category of the restaurant;
- order_protocol -> The protocol used to make the order;
- total_items -> Total items in the order; 
- subtotal -> Total value of the order in dollar cents;
- num_distinct_items -> Number os distinct items included in the order;
- min_item_price -> Price of the item with the least cost in the order in dollar cents;
- max_item_price -> Price of the item with the highest cost in the order in dollar cents;
- total_onshift_dashers -> Number of available dashers who are withing 10 miles of the store;
- total_busy_dashers -> Number of the total dashers who are withing 10 miles of the store and already working;
- total_outstanding_orders -> Number of orders wihtin 10 miles of this order that are currently being processed;
- estimated_order_place_duration -> Estimated order place duration given by other model in seconds;
- estimated_store_to_consumer_driving_duration -> Estimated time to the order get to the client after ready in seconds;

The objective is to predict the total delivery duration seconds which is defined as:
<center>total_delivery_duration_seconds = created_at - actual_delivery_time </center>  

## Create an application overview

The data_to_predict.json has request examples to test the application to be deployed and make live predictions. It can be run as a request or pass a json with the requests

The idea is to create a streamlit app to make customizable predictions and a FastAPI application.
