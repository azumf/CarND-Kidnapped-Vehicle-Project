/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;



void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	default_random_engine gen;
	// Number of particles
	num_particles = 50;
	// Init
	double std_x, std_y, std_theta;
	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];
	// Normal distributions for noise
	normal_distribution<double> dist_x(0, std_x);
	normal_distribution<double> dist_y(0, std_y);
	normal_distribution<double> dist_theta(0, std_theta);

	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = x;
		p.y = y;
		p.theta = theta;
		p.weight = 1.0;
		// Add some noise
		p.x += dist_x(gen);
		p.y += dist_y(gen);
		p.theta += dist_theta(gen);
		particles.push_back(p);
		weights.push_back(p.weight);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	default_random_engine gen;

	double std_x, std_y, std_theta;
	std_x = std_pos[0];
	std_y = std_pos[1];
	std_theta = std_pos[2];
	double x_mean, y_mean, theta_mean;
	// Normal distributions for noise
	normal_distribution<double> dist_x(0, std_x);
	normal_distribution<double> dist_y(0, std_y);
	normal_distribution<double> dist_theta(0, std_theta);

	for (int i = 0; i < num_particles; i++) {
		if (fabs(yaw_rate) < 0.0001) {
			particles[i].x += velocity*delta_t*cos(particles[i].theta);
			particles[i].y += velocity*delta_t*sin(particles[i].theta);
		}
		else {
			particles[i].x += velocity/yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += velocity/yaw_rate * (cos(particles[i].theta) - cos((particles[i].theta + yaw_rate * delta_t)));
			particles[i].theta += yaw_rate * delta_t;
		}
		// Add gaussian noise
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i = 0; i < observations.size(); i++) {
		LandmarkObs o = observations[i];
		// Init min_dist with numeric max limit for data type double
		double min_dist = numeric_limits<double>::max();
		int obj_id = -1;
		vector<double> distances;
		for (unsigned int j = 0; j < predicted.size(); j++) {
			LandmarkObs pred_ = predicted[j];
			double dist_ = dist(o.x, o.y, pred_.x, pred_.y);
			distances.push_back(dist_);
			if (dist_ < min_dist) {
				min_dist = dist_;
				obj_id = pred_.id;
			}
		}
		observations[i].id = obj_id;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i = 0; i < num_particles; i++) {
		double px_ = particles[i].x;
		double py_ = particles[i].y;
		double theta_ = particles[i].theta;

		// Vector to hold landmakrs
		vector<LandmarkObs> landmarks_in_range;

		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			
			// Get coordinates
			float LMx_ = map_landmarks.landmark_list[j].x_f;
			float LMy_ = map_landmarks.landmark_list[j].y_f;
			int LMid_ = map_landmarks.landmark_list[j].id_i;
			LandmarkObs current_landmark = { LMid_, LMx_, LMy_ };
			// Check if landmark is in sensor range
			if (fabs(dist(LMx_, LMy_, px_, py_)) <= sensor_range) {
				landmarks_in_range.push_back(current_landmark);
			}
		}
		
		// Vector to store transformed observations
		vector<LandmarkObs> obs_transformed;
		
		// Transform observations
		for (unsigned int l = 0; l < observations.size(); l++) {
			double trans_x = px_ + (cos(theta_) * observations[l].x) - (sin(theta_) * observations[l].y);
			double trans_y = py_ + (sin(theta_) * observations[l].x) + (cos(theta_) * observations[l].y);
			LandmarkObs transformed_obs;
			transformed_obs.id = observations[l].id;
			transformed_obs.x = trans_x;
			transformed_obs.y = trans_y;
			obs_transformed.push_back(transformed_obs);
		}

		// Data association (landmarks in range, transformed)
		dataAssociation(landmarks_in_range, obs_transformed);
		// Reset weight of particles[i].... to 1.0
		weights[i] = 1.0;
		double total_weight = 1.0;

		vector<int> associations_vector;
		vector<double> sense_x_vector;
		vector<double> sense_y_vector;

		// Get x & y of transformed observations
		for (unsigned int l = 0; l < obs_transformed.size(); l++) {
			double xtobs_, ytobs_, pr_x, pr_y;
			xtobs_ = obs_transformed[l].x;
			ytobs_ = obs_transformed[l].y;
			// Set variable = transformed id
			int id_obs = obs_transformed[l].id;

			for (unsigned int k = 0; k < landmarks_in_range.size(); k++) {
				if (landmarks_in_range[k].id == id_obs) {
					pr_x = landmarks_in_range[k].x;
					pr_y = landmarks_in_range[k].y;
				}
			}
			// Calculate weight with x,y of landmark in range
			double sx_ = std_landmark[0];
			double sy_ = std_landmark[1];
			//double expnt = ((pr_x - xtobs_)*(pr_x - xtobs_)) / (2 * sx_*sx_) + ((pr_y - ytobs_)*(pr_y - ytobs_) / (2 * sy_*sy_));
			//double obs_weight = 1 / (2 * M_PI * sx_ * sy_) * exp(-1 * expnt);
			// Combine both together in 1 step...
			double obs_weight = (1 / (2 * M_PI*sx_*sy_)) * exp(-(pow(pr_x - xtobs_, 2) / (2 * pow(sx_, 2)) +
				pow(pr_y - ytobs_, 2) / (2 * pow(sy_, 2)) - (2 * (pr_x - xtobs_)*(pr_y - ytobs_) / (sqrt(sx_)*sqrt(sy_)))));

			total_weight *= obs_weight;
			associations_vector.push_back(id_obs);
			sense_x_vector.push_back(xtobs_);
			sense_x_vector.push_back(ytobs_);
		}
		particles[i].weight = total_weight;

		SetAssociations(particles[i], associations_vector, sense_x_vector, sense_y_vector);
		landmarks_in_range.clear();
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	
	vector<Particle> resampled_particles;
	vector<double> weights;
	// Create a generator to be used for generating random particle index and beta value
	default_random_engine gen2;

	// Generate random particle index
	uniform_int_distribution<int> particle_index(0, num_particles - 1);

	int current_index = particle_index(gen2);

	double beta = 0.0;

	//double max_weight_2 = 2.0 * *max_element(weights.begin(), weights.end());
	double max_weight_2 = numeric_limits<double>::min();
	for (int i = 0; i < num_particles; i++) {
		weights.push_back(particles[i].weight);
		if (particles[i].weight > max_weight_2) {
			max_weight_2 = particles[i].weight;
		}
	}
	uniform_real_distribution<double> random_weight(0.0, max_weight_2);
	// Resampling wheel
	for (int i = 0; i < particles.size(); i++) {
		beta += random_weight(gen2);
		while (beta > weights[current_index]) {
			beta -= weights[current_index];
			current_index = (current_index + 1) % num_particles;
		}
		resampled_particles.push_back(particles[current_index]);
	}
	particles = resampled_particles;

}


Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

	// dClear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
