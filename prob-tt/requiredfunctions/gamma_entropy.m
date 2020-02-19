function entropy = gamma_entropy(a,b)
    if size(b,2) > 1 
        entropy = bsxfun(@minus,(gammaln(a)-(a-1).*psi(a)+a)',log(b));
    else
        entropy = gammaln(a)-(a-1).*psi(a)-log(b)+a;
    end
   %entropy = gammaln(a)-a*psi(a)-log(b)+a;
end