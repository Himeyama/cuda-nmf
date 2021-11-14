# frozen_string_literal: true

require_relative 'nmf/version'
require 'numo/narray'
require 'cuda/nmf.so'

module Cuda
  module NMF
    class Error < StandardError; end

    # NMF class
    class NMF
      def initialize(data, n_components, eps = 1e-4)
        # x: Numo::<T>Float
        # n_components: Integet
        # eps: Float
        data_class = data.class
        @m, @n, @k = data.shape + [n_components]
        @x = data
        @w = data_class.zeros(@m, @k)
        @h = data_class.zeros(@k, @n)
        @y = data_class.zeros(@m, @n)
        @eps = eps
        _NMF(@x, @w, @h, @y, @m, @n, @k, @eps)
      end
    end
  end
end
