set_languages("cxx11")

target("antdnn")
    set_kind("shared")
    add_defines("_DLL_ANTDNN")
    if is_plat("mingw") then
    	add_cxxflags("-fopenmp")
    	add_shflags("-fopenmp")
    else
    	add_cxxflags("/O2 /openmp /EHsc")
	end
    add_includedirs("include/antdnn")
    -- add_vectorexts("sse2", "sse3", "ssse3","avx","avx2", "mmx")
    add_includedirs("3rdparty")
    add_files("modules/*.cpp")

target("test")
	set_kind("binary")
	add_deps("antdnn")
	add_files("unit_test/test.cpp")
	add_includedirs("include")
	add_linkdirs("build")
	add_links("antdnn")