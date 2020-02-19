function [f,V,lF]=hyperg(v,z,opt);
% function [f,V,lF]=hyperg(v,z,opt);
% function evaluates properties of _0F_1(v,1/4 diag(D^2))
%     f	= dlogF/dz
%     V	= d2 logF/dz2
%     lF	= logF
%
% opt == 1       Mardia 1977
%        2       Bessel/Bessel
%        3       Woods 2002 (calibrated) [default]
%        4       Woods 2002 (non-calibrated)
%        5       Bessel/Bessel (with decreasing v)
% lF and V available for opt= 2,3,5 only!
%
% This function originates from,
% Vaclav vSmidl and Anthony Quinn, “On bayesian principal component
% analysis,” Computational statistics & data analysis, vol. 51, no. 9, pp.
% 4101–4123, 2007  
%

[r,s]	= size(z);
if nargin<3, opt=3; end;

switch opt
case 1	% Mardia 77
	on	= ones(r,1);
	z	= max([z on./v],[],2);
	sms	= 1./(on*z' + z*on');
	sms	= sms - diag(diag(sms));

	% hack for small z
	f	= on - 0.5*sum(sms)' - 0.5*(2*v-r)./z;
	f	= f.*(f>0);

case 2	% Besseli(v,z)/Besseli(v-1,z)
	n   = 100;
	pk  = 0;
	for k=n:-1:0
	    pk  = 1/4*z.^2/ ((v+k)*(v+k+1)) ./ (1 + pk);
	end

	f   = 1/2*z/v./(1+pk);

	% 100 iteration is not enough for z >> v, (e.g. 10x)
	% value is close to one anyway, using approx:

	ind = find(z>10*v);
	dD  = z(ind);
	d   = 2*v;
%	deS	= 1- exp(log((d-1)/2)-log(dD)-(d-3)/4./dD);
	deS	= 1- (d-1)/2./dD./exp((d-3)/4./dD);

	f(ind)  = deS;

	% approx
	if nargout>2
		vlF	= log(besseli(v-1,z)) + (v-1)*(log(2) - log(z)) + gammaln(v);
		lF	= sum(vlF);
	end;
	if nargout>1
		B2	= hyperg(v+1, z, opt);
		V	= B2.*f - f.^2 + f./z;
	end
case 3	% Woods - calibrated
	sqr	= sqrt(z.^2 + v^2);
	y	= z./(sqr + v);
	dy	= v./(sqr.*(sqr+v));

	y1m	= (1-y.^2 + 1e-20);
	yy	= 1./(1 - y.^2 * (y.^2)' + 1e-20); 	% matrix
	R0	= (diag(y) * (yy + diag(diag(yy))) * diag(y.^2)) * dy;

	%f	= R0 + y + z.*dy - v*2*y.*dy./y1m;
	f	= R0 + y + z.*dy - v*2*y.*dy./y1m;
	ind	= find(f>1);
	if ~isempty(ind),
		% use approx #5 RECURSIVE!
		f(ind)	= hyperg(v,z(ind),5);
	end;
	if any(f<0) | any(isnan(f)), keyboard; end

	if nargout>1 % evaluating covariance matrix
		d2y	= -(v + 2*sqr)*v.*z ./ (sqr.^3.*(sqr+v).^2);
		df	= -2*v*(dy.^2+y.*d2y)./y1m -4*v*y.^2.*dy.^2./y1m.^2 ...
			  +2*dy+z.*d2y + (dy.^2.*y.^2 + y.^3.*d2y)./diag(yy)...
			  +2*y.^6.*dy.^2./diag(yy).^2;
		%V	= diag(df);

		Vij	= (y*y') .* (dy*dy') .* yy + ...
			2*(y.^3*(y.^3)') .* (dy*dy') .* (yy.^2);
		V	= Vij - diag(diag(Vij)) + diag(df);
		if nargout>2
			lF	= sum(v*log(y1m)) + y'*z ...
				  +1/2*sum(sum(log(yy))+log(diag(yy))')/2;
		end
	end

case 4	% Woods - non callibrated
	sqr	= sqrt(z.^2 + v^2);
	y	= z./(sqr + v);
	dy	= v./(sqr.*(sqr+v));

	yy	= 1./(1 - y.^2 * (y.^2)' + 1e-20); 	% matrix
	R0	= (diag(y) * (yy + diag(diag(yy))) * diag(y.^2)) * dy ...
		  - y./(1+y.^2);

	f	= R0 + y + z.*dy - (v-0.5)*2*y.*dy./(1-y.^2 + 1e-20);
	ind	= find(f>1);
	if ~isempty(ind),	f(ind)	= 1; end;
	if any(f<0) | any(isnan(f)), keyboard; end

case 5	% Besseli(v,z)/Besseli(v-1,z) % with decreasing v=[v,v-1,...,1];
	n   = 100;
	pk  = 0;

	% decrease in dimensions
	orig_v	= v;
	v	= (v:-0.5:v-size(z,1)/2+0.5)';
	%

	for k=n:-1:0
	    pk  = 1/4*z.^2./ ((v+k).*(v+k+1)) ./ (1 + pk);
	end

	f   = 1/2*z./v./(1+pk);

	% 100 iteration is not enough for z >> v, (e.g. 10x)
	% value is close to one anyway, using approx:

	ind = find(z>10*v);
	dD  = z(ind);
	d   = 2*v(ind);
	%deS	= 1- exp(log((d-1)/2)-log(dD)-(d-3)/4./dD);
	deS	= 1- (d-1)/2./dD./exp((d-3)/4./dD);

	f(ind)  = deS;
	if nargout>1
		B2	= hyperg(orig_v+1, z, opt);
		V	= B2.*f - f.^2 + f./z;
	end
	if nargout>2
		ind	= find(z>1e-20);
		z	= z(ind);
		v	= v(ind); % 0F1 at zero is 1...
		%vlF2	= log(besseli(v-1,z)) + (v-1).*(log(2) - log(z)) + gammaln(v);
        vlF = log(besseli(v-1,z,1))+(abs(real(z))) + (v-1).*(log(2) - log(z)) + gammaln(v);
        %if any(isinf(vlF))
%            error('vlF had %i infinite values.',sum(isinf(vlF)))
            temp = log(besseli(v-1,z,1));
        %    temp(isinf(vlF)) = max(temp(~isinf(vlF)));
            vlF = temp+(abs(real(z))) + (v-1).*(log(2) - log(z)) + gammaln(v);
        %end
        
        %assert(all(~isinf(vlF)))
        %assert(all(abs(vlF-vlF2)./abs(vlF)<1e-12),'Note the two versions differ')
        %fprintf('vlF diff: %6.4e\n',max(abs(vlF-vlF2)./abs(vlF)))
		lF	= sum(vlF);
	end;
end
