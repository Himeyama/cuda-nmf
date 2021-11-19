#!/usr/bin/env ruby
# frozen_string_literal: true

# require 'bundler/setup'
require 'cuda/nmf'

x = Numo::DFloat[[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]]
nmf = Cuda::NMF::NMF.new(x, 2)
p nmf
# p nmf.rms
