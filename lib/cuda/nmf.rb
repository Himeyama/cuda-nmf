# frozen_string_literal: true

require_relative 'nmf/version'
require 'numo/narray'
require 'cuda/nmf.so'

module Cuda
  module NMF
    class Error < StandardError; end

    # NMF class
    class NMF
      def initialize(data, n_components, tol = 1e-4, max_iter = 200)
        # x: Numo::<T>Float
        # n_components: Integet
        # eps: Float
        data_class = data.class
        @m, @n, @k = data.shape + [n_components]
        @max_iter = max_iter
        @data = data
        @w = data_class.zeros(@m, @k)
        @h = data_class.zeros(@k, @n)
        @y = data_class.zeros(@m, @n)
        @e = data_class.zeros(@m, @n)
        @tol = tol
        _NMF(@data, @w, @h, @y, @e, @m, @n, @k, @tol)
      end

      def rms
        return @rms if @rms

        @rms = (@y - @data).square.sum
      end

      attr_accessor :data, :w, :h, :y, :e
      attr_reader :rss, :ss, :vaf, :m, :n, :k
    end
  end
end
