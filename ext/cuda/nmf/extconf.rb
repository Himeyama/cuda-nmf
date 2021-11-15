# frozen_string_literal: true

require 'mkmf'

# rubocop:disable Style/GlobalVars
$CXXFLAGS += ' -std=c++11'
# rubocop:enable Style/GlobalVars

cuda_dirs = Dir.glob('/usr/local/cuda-*').sort
raise SystemCallError, 2 if cuda_dirs.empty?

cuda_dir = cuda_dirs[0]
dir_config('cuda', "#{cuda_dir}/targets/x86_64-linux/include",
           "#{cuda_dir}/targets/x86_64-linux/lib/stubs")

narray = Gem::Specification.find_by_name('numo-narray')
raise 'Gem `numo-narray` not found' unless narray

narray_dir = narray.require_path
dir_config('numo-narray', "#{narray_dir}/numo", "#{narray_dir}/numo")
raise '`numo/narray.h` not found' unless have_header('numo/narray.h')

narray_lib = "#{narray_dir}/numo/narray.so"
raise '`narray.so` not found' unless File.exist? narray_lib

# rubocop:disable Style/GlobalVars
$libs += narray_lib if $libs.empty?
# rubocop:enable Style/GlobalVars

create_makefile('cuda/nmf') if have_header('cublas.h') && have_library('cublas')
