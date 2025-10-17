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

echo("---- TYPES AND OO STUFF -----");
// If you look at how our homebrew pairs and lists behave, they look a lot
// like datatypes (because they are!).
// We used functions as containers for data. In fact, we can do so arbitrarily:
some_complex_number = function(r, i) function(get_what) get_what == "real" ? r : i;
fiveplusfouri = some_complex_number(5, 4);
echo("real part", fiveplusfouri("real"));
echo("imaginary part", fiveplusfouri("imaginary"));

// We already used this pattern of "our original function returns a dispatcher function" above.
// We can also use the dispatching to return functions instead of values.
// But using strings is ugly, so let's first define some symbols:
p_real = 0;
p_img = 1;
m_add = 2;
m_sub = 3;
m_str = function(c) c(m_str);  // optional syntactic sugar for e.g. m_str(c1) instead of c1(m_str). But actually I prefer it the other way round (c1(m_str)), so we can chain commands in a more natural order later.
Complex = function(r, i) function(member)
      member == p_real ? r
    : member == p_img ? i
    : member == m_add ? function(c2) Complex(r+c2(p_real), i+c2(p_img))
    : member == m_sub ? function(c2) Complex(r-c2(p_real), i-c2(p_img))
    : member == m_str ? str(r, i > 0 ? "+" : "", i, "i")
    : assert(false, str("ERROR: UNKNOWN MEMBER ", member));

// Unfortunately, OpenSCAD does not support dot syntax. But the next best thing is parameter
// syntax, i.e. we provide the symbol for the method/property we want in parentheses:
//  c1(p_real) is equivalent to c1.p_real in a typical language
// In the case of methods, we then get the method and can call on arguments using another set of
// parentheses:
//  c1(m_add)(c2) == c1.add(c2) in another language
// Not extremely pretty, but much better than a bunch of nested parentheses!

c1 = Complex(3, 4);
c2 = Complex(2, 6);
echo("c1", c1(m_str));
echo("c1-c2", c1(m_sub)(c2)(m_str)); // equivalent to c1.m_sub(c2).m_str (method chaining)
echo("c1+c2", c1(m_add)(c2)(m_str));
// creating an error:
//echo(c1("hello"));

// This even supports inheritance - simply add/override what is needed and forward the rest to "super".
// We also add a "this" reference while we're at it (we assign our dispatching function to a
// local variable "this" so we can refer to it in our code, and then we return "this").
m_mult = 4;
Cplx2 = function(r, i)
  let (super = Complex(r, i), this = function(member)
      member == m_str ? str("Cplx2(", r, ", ", i, ")")
    : member == m_mult ? function(c2) Cplx2(r*c2(p_real)-i*c2(p_img), r*c2(p_img)+i*c2(p_real))
    : member == "thistestfoo" ? str(this(m_str), " thinks therefore it is.")
    : super(member)
   ) this;

c3 = Cplx2(3, 4);
c4 = Cplx2(2, 6);
echo("c3 (overridden)", c3(m_str));
echo("c3-c4 (inherited)", c3(m_sub)(c4)(m_str)); // equivalent to c1.m_sub(c2).m_str (method chaining)
echo("c3+c4 (inherited)", c3(m_add)(c4)(m_str));
echo("c3*c4 (new)", c3(m_mult)(c4)(m_str));
echo("c3(thistest) (new)", c3("thistestfoo"));

// All in all, this is actually surprisingly close to standard OO stuff. We even
// have a constructor (the outer function) which behaves exactly as a constructor should, and "this".
// We have chainable property/method selection and calls, no limit on parameter type or shape
// (no currying requirement), so this is not functional programming in disguise, but real
// object-oriented programming, with all the uglyness such as inheritance, that you should avoid. ;)

// There's one thing that we don't have, and that is mutable state.
// OpenSCAD does not offer mutable variables, which makes this kind of hard.
// In many cases, this is a non-issue, as any modifying method can simply return the
// modified object (which is in reality a new object, but in 90% of the cases, you won't
// care about the difference (unless you are addicted to hidden state changes, which would be worrying).
m_set_r = 5;
m_set_i = 6;
Cplx3 = function(r, i)
  let (super = Cplx2(r, i), this = function(member)
      member == m_set_r ? function(new_r) Cplx3(new_r, i)
    : member == m_set_i ? function(new_i) Cplx3(r, new_i)
    : super(member)
   ) this;

c5 = Cplx3(1,2);
echo("c5", c5(m_str));
c6 = c5(m_set_r)(42);
echo("c6", c6(m_str));

// But, alas, OpenSCAD does not allow different values for the same symbol.
// Except.... in loops (the incrementer) and recursion...
// So we COULD actually try to simulate mutability by using openscad's internal management
// of recusion to push our (changed or not) variable environment (or "stack frames") onto.
// Hmmmm...
// Maybe another time.


echo("---- MONAD STUFF -----");
// Drink a relaxing cup of tea while reading
// https://www.adit.io/posts/2013-04-17-functors,_applicatives,_and_monads_in_pictures.html
// and/or watching
// https://egghead.io/courses/professor-frisby-introduces-composable-functional-javascript

m_map = function(x) x(m_map);  // map/fmap/<$> (-> functor)
m_fold = function(x) x(m_fold);  // fold/reduce (-> functor)
m_bind = function(x) x(m_bind);  // bind/chain/>>=/flatMap/liftM (-> monad)
m_apply = function(x) x(m_apply);  // ap/<*>/liftA (-> applicative functor), requires currying or liftA tweaks to be effective
// not shown: of/unit/return/liftM (wrapping something in a monad is done through the constructor)
// BTW: bind + of = map
m_from_nullable = function(x) x(m_from_nullable);
m_from_num = function(x) x(m_from_num);
Maybe = function(val) function(member)  // todo: needed/useful at all?
      member == m_from_nullable ? function(x)  // something like a class method
                                     is_undef(x) ? Problem(x) : Result(x)
    : member == m_from_num ? function(x)
                                  is_undef(x) ? Problem(x)
                                : !is_num(x) ? Problem(str("Not a number: ", x))
                                : x == 1/0 ? Problem(x)
                                : x == -1/0 ? Problem(x)
                                : Result(x)
    : assert(false, "UNKNOWN MEMBER");
Result = function(val) let(super=Maybe(val)) function(member)
      member == m_map ? function(f) Result(f(val))  // wraps whatever value a function returns
    : member == m_fold ? function(resf, errf) resf(val)  // unwraps the value
    : member == m_bind ? function(f) f(val)  // returns whatever wrapped value a "monad maker" function returns
    : member == m_apply ? function(other) other.map(val)  // applies the function "val" to another Maybe
    : member == m_str ? str("Result(", val, ")")
    : super(member);
Problem = function(val) let(super=Maybe(val)) function(member)
      member == m_map ? function(f) Problem(val)
    : member == m_fold ? function(resf, errf) errf(val)
    : member == m_bind ? function(f) Problem(val)
    : member == m_apply ? function(other) Problem(val)
    : member == m_str ? str("Problem(", val, ")")
    : super(member);

echo(Result(3)(m_map)(function(x) x*2)
              (m_map)(function(x) x+7)
              (m_fold)(function(x) x, function(x) str("uh-oh: ", x)));

echo(Maybe()(m_from_num)(3)(m_str));
echo(Maybe()(m_from_num)(1/0)(m_str));
echo(Maybe()(m_from_num)(-1/0)(m_str));
echo(Maybe()(m_from_num)(undef)(m_str));
echo(Maybe()(m_from_num)("hello")(m_str));
echo(Maybe()(m_from_nullable)("hello")(m_str));
echo(Maybe()(m_from_nullable)(undef)(m_str));

// Maybe is a Functor if it satisfies two laws:
// 1. Preserve identity morphisms:
//    fx.map(id) == id(fx), where id = x => x
id = function(x) x;
echo("functor_neutral1", Result(3)(m_map)(id)(m_str));
echo("functor_neutral2", id(Result(3))(m_str));

// 2. Preserve composition of morphisms:
//    fx.map(f).map(g) == fx.map(x => g(f(x)))
plusthree = function(x) x+3;
timestwo = function(x) x*2;
functor_mapped = Result(2)(m_map)(plusthree)(m_map)(timestwo)(m_str);
functor_composed = Result(2)(m_map)(function(x) timestwo(plusthree(x)))(m_str);
echo("functor_mapped", functor_mapped);
echo("functor_composed", functor_composed);
echo("equal?", functor_mapped == functor_composed);

// Maybe is a Monad if it satisfies two more laws:
// 1. Left and right identity:
echo("monad left identity", Result(7)(m_bind)(plusthree));
echo("should equal", plusthree(7));
echo("monad right identity", Result(7)(m_bind)(Result)(m_str));
echo("should equal", Result(7)(m_str));

// 2. Associativity
f = function(x) Result(x+3);
g = function(x) Result(x*2);
echo("monad associativity left", Result(7)(m_bind)(f)(m_bind)(g)(m_str));
echo("monad associativity right", Result(7)(m_bind)(function(x) f(x)(m_bind)(g))(m_str));


// WIP: chaining binds with do notation
// (not really needed because of our nice method chaining)
do2 = function(m, f1, f2) m(m_map)(f1)(m_map)(f2);
echo("do2", do2(Result(2), timestwo, plusthree)(m_str));

do = function(m, f_vector) m(m_map)(reduce(compose, id, f_vector));
echo("do", do(Result(2), [plusthree, timestwo, inc])(m_str));
