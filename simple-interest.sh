#!/bin/bash

# Function to calculate simple interest
calculate_simple_interest() {
  local principal=$1
  local rate=$2
  local time=$3

  local interest=$(echo "scale=2; ($principal * $rate * $time) / 100" | bc)
  echo "The simple interest for the given values is: $interest"
}

# Read user input
read -p "Enter the principal amount: " principal
read -p "Enter the rate of interest: " rate
read -p "Enter the time (in years): " time

# Call the function with the user input
calculate_simple_interest $principal $rate $time
