# frozen_string_literal: true

require_relative 'lib/cuda/nmf/version'

Gem::Specification.new do |spec|
  spec.name          = 'cuda-nmf'
  spec.version       = Cuda::NMF::VERSION
  spec.authors       = ['Murata Mitsuharu']
  spec.email         = ['hikari.photon+dev@gmail.com']
  spec.summary       = 'GPU-based NMF for Ruby'
  spec.description   = 'NMF calculations are performed on NVIDIA GPUs using the Cuda API.'
  spec.required_ruby_version = '>= 2.5.0'
  spec.license = 'MIT'

  spec.files = Dir.chdir(File.expand_path(__dir__)) do
    `git ls-files -z`.split("\x0").reject { |f| f.match(%r{\A(?:test|spec|features)/}) }
  end
  spec.bindir        = 'exe'
  spec.executables   = spec.files.grep(%r{\Aexe/}) { |f| File.basename(f) }
  spec.require_paths = %w[lib ext]
  spec.extensions    = %w[ext/cuda/nmf/extconf.rb]
  spec.add_runtime_dependency 'numo-narray'
  spec.add_development_dependency 'rake-compiler'
end
