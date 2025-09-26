// Adding the map, filter, reduce functions to make it easier
// to do list/vector/array stuff in OpenSCAD.

// THE USEFUL FUNCTIONS TO COPY AND USE:

// needed plumbing for vectors (head: get first element, tail: all but first element)
head = function(vector) vector[0];
tail = function(vector) [ for (i=[1:1:len(vector)-1]) vector[i] ];

// map: vector --> vector with all elements transformed so that new_elem = func(elem)
map = function(func, vector) [ for(elem = vector) func(elem) ];

// filter: vector --> vector with fewer elements (skips elems if !func(elem))
filter = function(func, vector)
    len(vector) == 0 ?
        []
        : !func(head(vector)) ?
            filter(func, tail(vector))
            : concat([head(vector)], filter(func, tail(vector)));

// reduce: vector --> aggregated value (usually a number, but can be anything)
//         (func(prev_aggregate, elem) must return a new aggregate)
//         The original aggregate given should be the neutral element
//         (e.g. 0 for summation, 1 for addition, [] for concatenation, etc.).
//         Remember that search, min, max etc. are also aggregations and thus implementable using reduce.
reduce = function(func, aggregate, vector)
    len(vector) == 0 ?
        aggregate
        : reduce(func, func(head(vector), aggregate), tail(vector));

// Another potentially useful function: get the vector, but with index/value pairs
// e.g. for [a,b,c] it returns [[0,a], [1,b], [2,c]]
idx_val_pairs = function(vector) [for (i = [0:1:len(vector)-1]) [i, vector[i]]];


// EXAMPLE USAGE:

// example functions to demonstrate stuff with
add = function(a, b) a + b;
inc = function(x) x + 1;
double = function(x) x+x;
even = function(x) x % 2 == 0;

a = [3,4,2,1];//[1,2,3,4];
echo("a", a);

echo("map", map(double, a));

echo("filter", filter(even, a), "(even)");
echo("filter", filter(function(x) !even(x), a), "(not even, using anonymous function)");

echo("reduce", reduce(add, 0, a), "(sum)");

// Examples for solving your own problems with filter/map/reduce:

// Example 1: Getting the maximum value (max already exists, but hey)
my_max = function(vector)
    reduce(
        function(elem, aggr) elem > aggr ? elem : aggr,
        1/-0,  // -Inf
        vector);

echo("my_max", my_max(a));

// Example 2: Get the index of the maximum value (we start with elem 1, not 0 here,
// so that we can use 0 as a valid initial aggregate (index for the highest value))
argmax = function(vector)
    reduce(
        function(elem, aggr) elem[1] > vector[aggr] ? elem[0] : aggr,
        0,
        tail(idx_val_pairs(vector)));

echo("argmax", argmax(a));

// More generally, we can get the index of where an arbitrary predicate function 
// returns true (sort of like filter, but returning only the first matching element)
index_of = function(func, vector)
    reduce(
        function(elem, aggr) func(elem[1]) ? elem[0] : aggr,
        -1,
        idx_val_pairs(vector));

// This is more powerful than search() because it lets us use arbitrary tests and
// arbitrary data structures.
echo("index_of value 1", index_of(function(x) x==1, a));
// [Note, though, that this always iterates through all elements and returns the
// last match. If you want early stopping, you need to implement it as your
// own variant of reduce.]

// For example, we can use it to look up stuff in associative lists
num_legs = [
    ["dog", 4],
    ["duck", 2],
    ["long john silver", 1]
];
index_for_key = function(key, vector) index_of(function(x) x[0]==key, vector);
val_for_key = function(key, vector) vector[index_for_key(key, vector)][1];

echo("duck's index", index_for_key("duck", num_legs));
echo("duck's legs", val_for_key("duck", num_legs));

// Custom reduction example: Sum up all items up to and including "duck"
running_sum = function(key, vector)
    reduce(
        function(elem, aggr) 
            aggr[0] ? // has been found? --> just keep aggregrate value
                aggr
                : elem[0] == key ? // not yet found? --> check for match, then add
                    [true, aggr[1] + elem[1]] : [false, aggr[1] + elem[1]],
        [false, 0],  // aggr is a tuple of [has_been_found?, aggr_val]
        vector)[1]; // get sum's value from aggr tuple

echo("running_sum", running_sum("duck", num_legs));     

// Another way to do the same by combining stuff we already have.
// This shows how we can decompose the problem into relatively intuitive steps.
running_sum2 = function(key, vector)
    let (
        key_idx = index_for_key(key, vector),
        is_relevant = function(elem) elem[0] <= key_idx,
        relevant_elems = filter(is_relevant, idx_val_pairs(vector))
    )
    reduce(function(e, agg) e[1] + agg, 0, relevant_elems);

echo("running_sum2", running_sum("duck", num_legs));     


// ==================================
// OPTIONAL FUN AND POWERFUL STUFF

// We can also define map and filter in terms of the more general reduce,
// (more as a demonstration of its expressiveness than for other purposes)
map2 = function(func, vector)
    reduce(
        function(elem, aggregate) concat(aggregate, func(elem)),
        [],
        vector);

filter2 = function(func, vector)
    reduce(
        function(elem, aggregate) !func(elem) ? aggregate : concat(aggregate, elem),
        [],
        vector);
       

echo("map2", map2(double, a));
echo("filter2", filter2(even, a), "(even)");

// Composing functions without having to execute them immediately
compose = function(func1, func2) function(x) func1(func2(x));

not = function(x) !x;
noteven = compose(not, even);

echo("fun filter", filter(noteven, a), "(noteven)");

// Partial function application ("pre-configuring" functions, but not yet executing)
// We need different variants depending on num of arguments of the original function.
// If func expects N arguments, then the resulting function expects N-1 arguments.
partial2 = function(func, x) function(y) func(x, y);
partial3 = function(func, x) function(y, z) func(x, y, z);
partial4 = function(func, x) function(y, z, zz) func(x, y, z, zz);

even_filter = partial2(filter, even);
echo("fun even_filter", even_filter(a));

// try with more than 1 parameter:
sum = partial2(partial3(reduce, add), 0);
echo("fun sum", sum(a));

// hmm, ugly, we're nesting calls to partial... Better visible here:
bla = function(a, b, c) a+b*c;
bla1 = partial3(bla, 1);
bla2 = partial2(bla1, 2);
echo("bla2", bla2(3));

// More generically, we should use "currying" to transform a function so that, until
// we arrive at the last parameter, every single-parameter application returns
// a new function with one fewer argument.
//curry1 = function(func) function(x) func(x);  // same as func, not needed
curry2 = function(func) function(x) function(y) func(x, y);
curry3 = function(func) function(x) function(y) function(z) func(x, y, z);
curry4 = function(func) function(x) function(y) function(z) function(zz) func(x, y, z, zz);

// The transformation to a curried function is then just one function call using
// curry(...) of the right arity (right number of arguments).
blu1 = curry3(bla);
echo("blu1", blu1(1)(2)(3));

// So to revisit sum, we can use one call to curry insteod of two calls to partial:
sum2 = curry3(reduce)(add)(0);
echo("fun sum curried", sum2(a));

// Curried functions don't have many uses outside of partial application.
// But they are "clean" in that they always only expect one argument at any time.
// Therefore, they can be composed in a generic way (because composision assumes
// single-argument functions).

doubler = curry2(map)(double);
double_then_sum = compose(doubler, sum2);
echo("fun double_then_sum", double_then_sum(a));

// So we gave information to a function, but did not yet evaluate it.
// This is related to lazy evaluation. Lazy evaluation is useful in that it
// only causes work if/when the result is actually needed.
// If the parameters also come from functions, this creates a tree
// that only gets evaluated once we want to do something with its root node.
// This also makes it possible to "package" a piece of work beforehand, without
// having to pass around all its parameters as well. Also called a Thunk.
lazy_val = function(val) function() val;
lazy1 = function(func, x) function() func(x());
lazy2 = function(func, x, y) function() func(x(), y());
lazy3 = function(func, x, y, z) function() func(x(), y(), z());
lazy4 = function(func, x, y, z, zz) function() func(x(), y(), z(), zz());

expensive_add = function(x,y) echo("now adding") x+y;
expensive_mult = function(x,y) echo("now multiplying") x*y;

echo("nothing calculating yet:");
result = lazy2(expensive_mult, lazy_val(3),
                    lazy2(expensive_add, lazy_val(1), lazy_val(2)));
echo(result);

echo("calculating... nnnnn...");
echo(result(), "...NOW!");


// Now that we've veered off useful territory anyway, we can have a look at
// the fact that functions as such can actually replace data structures!
// So if we're dissatisfied with openscad's vectors, we can roll our own. ;)
// Let me show you:
pair = function(a, b) function(left) left ? a : b;
left = function(a_pair) a_pair(true);
right = function(a_pair) a_pair(false);

p = pair(1, 2);
echo(left(p), right(p));

// We can easily create lists from pairs (if you can excuse that openscad does
// not provide us a nice syntax to do this):
a_list = pair(1, pair(2, pair(3, pair(4, undef))));

echo("second elem:", left(right(a_list)));

// So 'left' is actually the same as 'head' and 'right' is the same as 'tail'.

// Implement indexing (without square brackets, but that's just syntactic sugar):
get_nth = function(l, idx, curr_idx=0)
    is_undef(l) ? undef
        : curr_idx == idx ? left(l)
            : get_nth(right(l), idx, curr_idx+1);

echo("second elem by index:", get_nth(a_list, 1));

// Okay, I'll cave: here's a converter from openscad vectors to our lists:
list = function(vector, curr_list=undef)
    len(vector) == 0 ? curr_list
        : list([for(i=[0:1:len(vector)-2]) vector[i]],
               pair(vector[len(vector)-1], curr_list));

vector = function(l, curr_vec=[])
    is_undef(l) ? curr_vec
        : vector(right(l),
                 concat(curr_vec, [left(l)]));

a_list2 = list([1,2,3,4]);
echo("third elem by index:", get_nth(a_list2, 2));
echo("a_list2 as vector", vector(a_list2));

lreduce = function(func, agg, l)
    is_undef(l) ? agg
        : lreduce(func, func(left(l), agg), right(l));

echo("lreduce sum", lreduce(add, 0, a_list2));

lmap = function(func, l)
    is_undef(l) ? undef
        : pair(func(left(l)), lmap(func, right(l)));

echo("lmap double", vector(lmap(double, a_list2)));

lfilter = function(func, l)
    is_undef(l) ? undef
        : func(left(l)) ? 
            pair(left(l), lfilter(func, right(l)))
            : lfilter(func, right(l));

echo("lfilter even", vector(lfilter(even, a_list2)));

lreverse = function(l) lreduce(
    function(elem, agg) pair(elem, agg),
    undef,
    l);

echo("lreverse", vector(lreverse(a_list)));

// Assoc lists:
num_legs2 = list([
    pair("dog", 4),
    pair("duck", 2),
    pair("long john silver", 1)
]);

get_assoc = function(key, al)
    is_undef(al) ? undef
        : left(left(al)) == key ? right(left(al))
            : get_assoc(key, right(al));

echo("duck has", get_assoc("duck", num_legs2), "legs");

sum_to = function(key, al, result=0)
    is_undef(al) ? undef
        : left(left(al)) == key ? result + right(left(al))
            : sum_to(key, right(al), result + right(left(al)));

echo("sum_to duck", sum_to("duck", num_legs2));

// Interesting post: https://stackoverflow.com/a/79620666
