# What does this project do?

We built a system that predicts how many people will enroll for Aadhaar in different parts of India.

## How does it work?
- It looks at past enrolments
- Learns patterns for each state and district
- Understands short-term trends (last week matters more than last year)
- Adjusts predictions when data is noisy or missing

## Why is it useful?
- Helps plan enrollment resources
- Identifies seasonal spikes early
- Works even when historical data is limited

## Why is this model reliable?
- It was tested month-by-month on unseen data
- It avoids overfitting automatically
- It explains *why* it makes predictions

## What makes it special?
- Automatically cleans messy state and district names
- Handles days with zero enrolments safely
- Shows confidence ranges, not just single numbers

In short:  
**It doesn’t just predict numbers. It explains them and knows when to be uncertain.**

